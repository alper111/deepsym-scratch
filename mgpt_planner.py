"""
Modified from below:
https://github.com/sd-james/skills-to-symbols/blob/master/s2s/planner/mgpt_planner.py
Original author: Steven James
Thanks Steven!
"""

import os
import subprocess
import numpy as np
from subprocess import Popen, PIPE
from time import sleep
from typing import Tuple, Any, List


class PlanOutput:

    def __init__(self, raw_output: str):
        self.valid = 'goal reached!' in raw_output
        self.raw_output = raw_output
        if self.valid:
            self.path = self._extract(raw_output)
        else:
            self.path = []

    def _extract(self, output):

        output = output[output.index('<begin-session>'):].split('\n')
        n_rounds = int(output[1][output[1].index('=') + 1:])
        row = 4
        # best_score = np.inf
        # best = list()
        plans = {}
        for _ in range(n_rounds):
            row, success, moves = self._extract_round(output, row)
            if success:
                curr_plan = "\n".join(moves)
                if curr_plan in plans:
                    plans[curr_plan] += 1
                else:
                    plans[curr_plan] = 1
            # if success and len(moves) < best_score:
            #     best_score = len(moves)
            #     best = moves
        best_count = 0
        best = None
        for key in plans:
            if plans[key] > best_count:
                best = str(plans[key]) + " times:\n" + key
                best_count = plans[key]
        return best

    def _extract_round(self, output, row):
        line = output[row].strip()
        moves = list()
        while line.startswith('+'):
            moves.append(line[2:-1])
            row += 1
            line = output[row].strip()
        success = 'goal reached!' in output[row]
        return row + 4, success, moves


class mGPT:
    """
    A class that wraps the mGPT planner with mdpsim.
    """

    def __init__(self,
                 mdpsim_path="mdpsim/mdpsim",
                 mgpt_path="mini-gpt/planner",
                 planner="lrtdp",
                 heuristic="ff",
                 port=2323,
                 max_time=10,
                 rounds=100):
        """
        Create a new PDDL planner object
        :param mgpt_path: the path to the mGPT executable
        :param planner: the type of planner to use. Default is lrtdp.
        :param heuristic: the heuristic to use. Default is ff.
        :param port: the port to run mdpsim on. It can be anything that is free
        :param max_time: the maximum time in seconds the planner is allowed. Default is 10
        :param kwargs: allows user to specify additional mGPT settings
        """
        if not os.path.exists(mgpt_path):
            raise ValueError("Could not find executable file at {}".format(mgpt_path))
        if not os.path.exists(mdpsim_path):
            raise ValueError("Could not find executable file at {}".format(mdpsim_path))

        self._mdpsim_port = port
        self._mdpsim_path = mdpsim_path
        self._planner_path = mgpt_path
        self._planner = planner
        self._heuristic = heuristic
        self._max_time = max_time
        self._rounds = rounds

    def find_plan(self, domain: Any, problem: Any) -> Tuple[bool, Any]:
        """
        Given a path to the PDDL domain and problem file, determine if a plan can be found. We do this by spinning up
        mdpsim, then running mGPT, then shutting it all down!
        :param domain: the domain file
        :param problem: the problem file
        :return: the first boolean represents whether the files were valid PDDL, the second represents whether a plan
        could be found and a list of output from the planner
        """

        # first, create a new file that has both the domain and problem in it.
        temp_name, problem_name = self._create_temp_file(domain, problem)

        p = None
        p2 = None

        try:
            print("Starting mdpsim...")
            # now run mdpsim with the temp file as input
            p = Popen(
                [self._mdpsim_path,
                 "--port",
                 "{}".format(self._mdpsim_port),
                 "--warnings=1",
                 '-R %d' % self._rounds,
                 temp_name
                 ],
                stdout=PIPE, stderr=PIPE, universal_newlines=True
            )

            # give it a second necessary?
            sleep(1)

            if p.poll():
                # it finished! something bad must have happened
                std_out, std_err = p.communicate()
                if p.returncode != 0:
                    print("mdpsim failed to start with error: {}".format(std_err))
                    return False, std_err

            print("mdpsim started!")
            print("Starting mGPT...")
            # it's running well! now start planner
            p2 = Popen(
                [self._planner_path,
                 '-v',
                 '100',
                 '-p',
                 self._planner,
                 '-h',
                 self._heuristic,
                 'localhost:{}'.format(self._mdpsim_port),
                 temp_name,
                 problem_name
                 ],
                stdout=PIPE, stderr=PIPE, universal_newlines=True
            )
            print("mGPT started! Waiting for output...")

            try:
                std_out, std_err = p2.communicate(timeout=self._max_time)
                output = PlanOutput(std_out)
                return True, output
            except subprocess.TimeoutExpired:
                return False, 'Timeout!'
        finally:
            os.remove(temp_name)
            os.remove("last_id")  # created by the planner and mdpsim
            os.system("rm -r logs")
            if p:
                p.terminate()
            if p2:
                p2.terminate()

    def _create_temp_file(self, domain: Any, problem: Any) -> Tuple[str, str]:

        name = "temp.pddl"
        print("Generating temp PDDL file {}".format(name))
        if isinstance(domain, str):

            # it's a path to a file!
            if not os.path.exists(domain):
                raise ValueError("Could not find PDDL file at {}".format(domain))
            if not os.path.exists(problem):
                raise ValueError("Could not find PDDL file at {}".format(problem))

            with open(name, 'w') as temp_file, open(domain, 'r') as domain_file, open(problem, 'r') as problem_file:
                temp_file.write('{}\n\n{}'.format(domain_file.read(), problem_file.read()))
            problem_name = self._extract_problem_name(problem)
            return name, problem_name

        else:

            # it's the objects!
            with open(name, 'w') as temp_file:
                temp_file.write('{}\n\n{}'.format(domain, problem))

            return name, problem.name

    def _extract_problem_name(self, problem_path: str) -> str:
        search = '(define (problem '
        with open(problem_path, 'r') as file:
            content = file.read().strip('\n')
            idx = content.index(search)
            end = content.index(')', idx + 1)
            name = content[idx + len(search):end]
            return name

    def extract_plan(self, output: List[str]) -> Tuple[float, List[str]]:
        start_idx = output.index('found plan as follows:')
        end_idx = [i for i, s in enumerate(output) if 'total cost' in s][0]
        plan = [' '.join(output[i].split(' ')[1:]) for i in range(start_idx + 1, end_idx)]
        cost = float(output[end_idx].split(' ')[-1])
        return cost, plan
