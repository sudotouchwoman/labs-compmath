from utils import uncertainty

if __name__ == '__main__':
    settings = uncertainty.load_config('res/config.json')
    uncertainty.perform_analysis(settings)