from pipeline import main

nlp_methods = ['tfidf']
fs_methods = ['chi_square', 'mutual_info']
model_methods = ['svm', 'rfc', 'nb']

for nlp in nlp_methods:
    for fs in fs_methods:
        for model in model_methods:
            print(f"Running {nlp} {fs} {model} pipeline")
            main(nlp, fs, model)
