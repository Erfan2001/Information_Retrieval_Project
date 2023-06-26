from src.classifier.classifier import MyClassifier

def tester():
    config = load_config_file()

    # Adjust seed values
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    API_URL = config.modelURL or "https://api-inference.huggingface.co/models/Erfan2001/zzz"
    token = config.token or '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'
    model = MyClassifier(config)
    model.inference(API_URL,token,os.path.join(config.data_dir,'test'),
                    os.path.join(config.cache_dir,'normalized_test'),
                    os.path.join(config.cache_dir,'tokenized_test'))

if __name__ == "__main__":
    tester()