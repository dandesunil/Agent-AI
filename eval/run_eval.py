from deepeval import DeepEval

def run():
    evaluator = DeepEval(config_path='deepeval_config.yml')
    results = evaluator.run()
    print('Evaluation Results:', results)

if __name__ == '__main__':
    run()
