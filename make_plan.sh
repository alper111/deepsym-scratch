#!/bin/bash

run_pddl() {
    # concatenate domain and problem file
    cat "$1/$2" > "$1/temp.pddl"
    cat "$1/problem.pddl" >> "$1/temp.pddl"
    # open PDDL server
    ./mdpsim/mdpsim --port=2322 -R 100 --time-limit=10000 "$1/temp.pddl" & _pid="$!"
    # save mdpsim pid
    echo "$_pid" > server.pid
    # run planner and save temporary result to planresult.txt
    # to see other planning options, run the planner without any argument
    # e.g. ./mini-gpt/planner
    ./mini-gpt/planner -v 100 -h ff localhost:2322 "$1/temp.pddl" dom1 > "$1/planresult.txt"
    # kill mdpsim server
    kill -9 "$(cat server.pid)"
    # remove auxilliary files
    rm server.pid
    rm -r logs

    # parse the plan result
    # see the plan in {savepath}/plan.txt
    python parse_plan.py -opts "$1/opts.yaml"
    mv "$1/plan.txt" "$1/$3"
    rm "$1/planresult.txt"
}

concat_to_first() {
    cat "$2" > "temp.txt"
    cat "$1" >> "temp.txt"
    rm "$1"
    mv "temp.txt" "$1"
}

# get save location
loc="$(grep save: $1 | sed 's/^.*: //')"

# # generate a scene
# python generate_scene.py
# # transform image to pddl problem
# python recognize.py -opts "$1" -goal "$2"

run_pddl "$loc" "ddomain.pddl" "pddl-plan.txt"
run_pddl "$loc" "pdomain.pddl" "ppddl-plan.txt"

python mcts_plan.py -opts "$1" -goal "$2" > "$loc/mcts-plan.txt"

echo "PDDL plan"
cat "$loc/pddl-plan.txt"
echo "PPDDL plan"
cat "$loc/ppddl-plan.txt"
echo "MCTS plan"
cat "$loc/mcts-plan.txt"

concat_to_first "$loc/pddl-plan.txt" "$loc/objects.txt"
concat_to_first "$loc/ppddl-plan.txt" "$loc/objects.txt"
concat_to_first "$loc/mcts-plan.txt" "$loc/objects.txt"
