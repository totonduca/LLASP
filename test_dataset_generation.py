import numpy as np
import pandas as pd
import fine_tuning

###############################################

###     FOR PROMPT INVARIANCE GENERATE OTHER 20 PROMPTS E CHOOSE RANDOMICALLY ONE OF THEM EVERY CYCLE

engineering = f'''You are an expert in Answer Set Programming (ASP). Your task is to generate correct ASP programs based on problem descriptions.  

## ASP Rules:
- An **atom** is of the form `p(t1, ..., tn)`, where `p` is a predicate and `t1, ..., tn` are terms.
- A **rule** consists of a head and a body: `head :- body.`  
- A **constraint** prevents certain solutions: `:- condition.`  
- Comments start with `%` and explain the logic.

**Problem:** "Find the blue nodes of the graph." **ASP Code:**
``` blue_nodes(X):- node(X), color(X, "blue"). ```
'''

#################       PROVARE CAMBIANDO L'ESEMPIO FORNITO DI BASE, E POI ANCHE SENZA ESEMPIO, SU TUTTI E TRE I MODELLI


def build_test_set():
    seed = 721345631                      #############       PRIMA 1000 MA ORA RIDUCIAMO

    colors = ["pink", "white", "black", "darkmagenta", "lightblue"]
    cities = ["cosenza", "delhi", "cairo", "mumbai", "moscow", "singapore", "chicago", "toronto", "barcelona"]
    labels = ["wall", "chair", "roof", "flower", "butterfly", "laptop", "desk", "cloud", "storm"]
    attributes = ["surname", "owner", "lake", "hair", "weight", "strength", "quality"]

    predicates = colors + cities + labels + attributes
    closures = ["loops", "family", "trains", "journey"]

    test_tuples = []

    match testing:
        case "core":
            for i in range(n_size):
                np.random.seed(seed % (i + 1) * 19)
                
                chosen = 0

                questions_assignments, answers_assignments, facts_assignments = fine_tuning.label_assignment(predicates, np.random.choice(predicates), False, False)

                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[chosen]])

                questions_prevents, answers_prevents, facts_prevents = fine_tuning.prevent_value(predicates, np.random.choice(predicates), False, False)

                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = fine_tuning.generate_combinations(p_1, p_2, False, False)

                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = fine_tuning.execute_join(p_1, p_2, attributes, False, False)

                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = fine_tuning.transitive_closure(np.random.choice(closures), np.random.choice(predicates), False, False)

                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[chosen]])

                questions_preferences, answers_preferences, facts_preferences = fine_tuning.preferences(np.random.choice(predicates),predicates, False, False)

                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[chosen]])

                questions_filtering, answers_filtering, facts_filtering = fine_tuning.select_value(np.random.choice(predicates), np.random.choice(predicates), False, False)

                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[chosen]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = fine_tuning.select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, False, False)

                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[chosen]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = fine_tuning.select_by_numeric_condition(np.random.choice(predicates), False, False)

                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[chosen]])
            
        case "invariance":
            prompt_invariance=True

            for i in range(n_size):
                np.random.seed(seed % (i + 1) * 19)

                questions_assignments, answers_assignments, facts_assignments = fine_tuning.label_assignment(predicates, np.random.choice(predicates), prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[0]])

                questions_prevents, answers_prevents, facts_prevents = fine_tuning.prevent_value(predicates, np.random.choice(predicates), prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = fine_tuning.generate_combinations(p_1, p_2, prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = fine_tuning.execute_join(p_1, p_2, attributes, prompt_invariance, False)

                chosen = np.random.randint(0, 20*9)             ###########     20*9 perché ogni chiamata a join genera 9 tipi di join e 20 prompt per ogni tipo
                
                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = fine_tuning.transitive_closure(np.random.choice(closures), np.random.choice(predicates), prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[0]])

                questions_preferences, answers_preferences, facts_preferences = fine_tuning.preferences(np.random.choice(predicates),predicates, prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[0]])

                questions_filtering, answers_filtering, facts_filtering = fine_tuning.select_value(np.random.choice(predicates), np.random.choice(predicates), prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[0]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = fine_tuning.select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[0]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = fine_tuning.select_by_numeric_condition(np.random.choice(predicates), prompt_invariance, False)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[0]])
    
        case "core-invariance":
            prompt_invariance=True

            for i in range(n_size):
                np.random.seed(seed % (i + 1) * 19)
                
                chosen = 0

                questions_assignments, answers_assignments, facts_assignments = fine_tuning.label_assignment(predicates, np.random.choice(predicates), not prompt_invariance, False)

                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[chosen]])

                questions_prevents, answers_prevents, facts_prevents = fine_tuning.prevent_value(predicates, np.random.choice(predicates), not prompt_invariance, False)

                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = fine_tuning.generate_combinations(p_1, p_2, not prompt_invariance, False)

                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = fine_tuning.execute_join(p_1, p_2, attributes, not prompt_invariance, False)

                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = fine_tuning.transitive_closure(np.random.choice(closures), np.random.choice(predicates), not prompt_invariance, False)

                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[chosen]])

                questions_preferences, answers_preferences, facts_preferences = fine_tuning.preferences(np.random.choice(predicates),predicates, not prompt_invariance, False)

                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[chosen]])

                questions_filtering, answers_filtering, facts_filtering = fine_tuning.select_value(np.random.choice(predicates), np.random.choice(predicates), not prompt_invariance, False)

                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[chosen]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = fine_tuning.select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, not prompt_invariance, False)

                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[chosen]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = fine_tuning.select_by_numeric_condition(np.random.choice(predicates), not prompt_invariance, False)

                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[chosen]])
            
                ## start invariance

                questions_assignments, answers_assignments, facts_assignments = fine_tuning.label_assignment(predicates, np.random.choice(predicates), prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_assignments[chosen], answers_assignments[chosen], facts_assignments[0]])

                questions_prevents, answers_prevents, facts_prevents = fine_tuning.prevent_value(predicates, np.random.choice(predicates), prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_prevents[chosen], answers_prevents[chosen], facts_prevents[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_combinations, answers_combinations, facts_combinations = fine_tuning.generate_combinations(p_1, p_2, prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_combinations[chosen], answers_combinations[chosen], facts_combinations[0]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_join, answers_join, facts_join = fine_tuning.execute_join(p_1, p_2, attributes, prompt_invariance, True)

                chosen = np.random.randint(0, 20*9)             ###########     20*9 perché ogni chiamata a join genera 9 tipi di join e 20 prompt per ogni tipo
                
                test_tuples.append([questions_join[chosen], answers_join[chosen], facts_join[chosen]])

                questions_closure, answers_closure, facts_closure = fine_tuning.transitive_closure(np.random.choice(closures), np.random.choice(predicates), prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_closure[chosen], answers_closure[chosen], facts_closure[0]])

                questions_preferences, answers_preferences, facts_preferences = fine_tuning.preferences(np.random.choice(predicates),predicates, prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_preferences[chosen], answers_preferences[chosen], facts_preferences[0]])

                questions_filtering, answers_filtering, facts_filtering = fine_tuning.select_value(np.random.choice(predicates), np.random.choice(predicates), prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_filtering[chosen], answers_filtering[chosen], facts_filtering[0]])

                questions_negative_filtering, answers_negative_filtering, facts_negative_filtering = fine_tuning.select_by_negative_condition(np.random.choice(predicates), np.random.choice(predicates), predicates, prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_negative_filtering[chosen], answers_negative_filtering[chosen], facts_negative_filtering[0]])

                questions_numeric_filtering, answers_numeric_filtering, facts_numeric_filtering = fine_tuning.select_by_numeric_condition(np.random.choice(predicates), prompt_invariance, True)

                chosen = np.random.randint(0, 20)
                test_tuples.append([questions_numeric_filtering[chosen], answers_numeric_filtering[chosen], facts_numeric_filtering[0]])
    
        case "comples":
            for i in range(n_size):
                np.random.seed(seed % (i + 1) * 19)
                
                chosen = 0
                
                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_jnf, answers_jnf, facts_jnf = fine_tuning.join_numeric_filtering(p_1, p_2, attributes)

                test_tuples.append([questions_jnf[chosen], answers_jnf[chosen], facts_jnf[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_jneg, answers_jneg, facts_jneg = fine_tuning.join_filtering(p_1, p_2, attributes, labels)

                test_tuples.append([questions_jneg[chosen], answers_jneg[chosen], facts_jneg[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_cg, answers_cg, facts_cg = fine_tuning.closure_guessing(labels, p_1, p_2)

                test_tuples.append([questions_cg[chosen], answers_cg[chosen], facts_cg[chosen]])

                questions_gc, answers_gc, facts_gc = fine_tuning.guessing_constraint(labels, np.random.choice(predicates))

                test_tuples.append([questions_gc[chosen], answers_gc[chosen], facts_gc[chosen]])

                questions_gnf, answers_gnf, facts_gnf = fine_tuning.guessing_numeric_filtering(labels, np.random.choice(predicates), np.random.choice(attributes), np.random.choice(attributes))

                test_tuples.append([questions_gnf[chosen], answers_gnf[chosen], facts_gnf[chosen]])

                questions_gf, answers_gf, facts_gf = fine_tuning.guessing_filtering(labels, np.random.choice(predicates))

                test_tuples.append([questions_gf[chosen], answers_gf[chosen], facts_gf[chosen]])

                p_1, p_2 = np.random.choice(predicates, 2, replace=False)
                questions_cnf, answers_cnf, facts_cnf = fine_tuning.combination_constraint(np.random.choice(labels), p_1, p_2)

                test_tuples.append([questions_cnf[chosen], answers_cnf[chosen], facts_cnf[chosen]])

    return test_tuples


###   CHOOSE BETWEEN [' core ' , ' invariance ' , ' core-invariance ' , ' comples ']

testing = "comples"

match testing:
    case "core":
        print("core")
        file_name = "data/test_core.csv"
        n_size = 1000   
    case "invariance":
        file_name = "data/test_invariance.csv"
        n_size = 1000    
    case "core-invariance":
        print("core_invariance")
        file_name = "data/test_core_invariance.csv"
        n_size = 500
    case "comples":
        print("comples")
        file_name = "data/test_comples.csv"
        n_size = 1000


# if prompt_engineering:
#     file_name = "data/test_core_invariance_prompt_engineering_simple.csv"
    
test_tuples = build_test_set()

test_df = pd.DataFrame(test_tuples, columns=["prompt", "answer", "fact"])

# CHANGE PATH IF NEEDED
test_df.to_csv(file_name, index=False)