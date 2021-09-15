from utils import uncertainty

if __name__ == '__main__':
    h_settings = uncertainty.load_config('res/Y-config.json')
    uncertainty.H_analysis(h_settings)

    x_settings = uncertainty.load_config('res/X-config.json')
    uncertainty.X_analysis(x_settings)
    