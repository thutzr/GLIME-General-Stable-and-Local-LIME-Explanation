import numpy as np
from PIL import Image
import time
import os
import pandas as pd 
import torch 
from torchvision.transforms import transforms
import torch.nn.functional as F
from lime import lime_image, lime_text, lime_tabular
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.linear_model import Ridge, Lasso

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

def get_pixel_explanation(explanation, label, num_features=10):
    segments = explanation.segments
    img = explanation.image
    exp = explanation.local_exp[label]
    attrs = np.zeros((img.shape[:2]))
    for f,w in exp[:num_features]:
        attrs[segments == f] = w

    return attrs  


def lime_explanation(model, x, device, random_seed=2023, n_samples=100,
                     kernel_width=3, distribution='exp', weighted=False,
                     distance_metric='l2',
                     alpha=1):
    set_seed(random_seed)
    model.eval()
    probs = model(x.to(device))
    probs = F.softmax(probs, dim=1)
    probs = probs[0].detach().cpu().numpy()
    label = probs.argmax().item()
    explainer = lime_image.LimeImageExplainer(kernel_width=kernel_width,verbose=False, random_state=random_seed)
    if distribution in ['smooth_grad_l1', 'smooth_grad']:
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                        max_dist=0, ratio=0.2,
                                                        random_seed=2023)
    else:
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                        max_dist=200, ratio=0.2,
                                                        random_seed=2023)
    
    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(get_preprocess_transform()(i) for i in images), dim=0)

        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    x = np.array(Image.fromarray(np.rollaxis(np.uint8(x.cpu().squeeze().numpy()*255), 0, 3)))
    segments = segmenter(x)
    
    regressor = 'l2'
    if distribution in ['comb_exp_l1', 'gaussian_additive_l1', 'smooth_grad_l1']:
        regressor = 'l1'
        distribution = '_'.join(distribution.split('_')[:-1])
        
    if regressor == 'l2':
        if distribution == 'smooth_grad':
            alpha = 0
        model_regressor = Ridge(alpha=alpha, fit_intercept=True,
                                    random_state=random_seed)
    elif regressor == 'l1':
        alpha = 1/(2*n_samples)
        model_regressor = Lasso(alpha=alpha, fit_intercept=True,
                                    random_state=random_seed)
        
    if distribution in ['uniform', 'uniform_adaptive']:
        weighted = True
    
    if distribution != 'random':
        explanation = explainer.explain_instance(x, 
                                            batch_predict, # classification function
                                            segmentation_fn=segmenter,
                                            hide_color=0, 
                                            distance_metric=distance_metric,
                                            top_labels=None,
                                            labels=(label,),
                                            num_samples=n_samples,
                                            batch_size=128,
                                            model_regressor=model_regressor,
                                            distribution=distribution,
                                            weighted=weighted, 
                                            random_seed=random_seed,
                                            model=model)
        

        return {'local_exp':explanation.local_exp[label], 
                'intercept':explanation.intercept[label],
                'score':explanation.score[label],
                'local_pred':explanation.local_pred[label],
                'true_prob':probs[label]}
    else:
        segments = segmenter(x)
        explanation = np.random.rand(len(np.unique(segments)))
        
        return explanation


def lime_text_explanation(model, tokenizer, x, device, random_seed=2023, n_samples=100,
                     kernel_width=3, distribution='exp', weighted=False,
                     distance_metric='cosine',
                     alpha=1):
    set_seed(random_seed)
    model.eval()
    probs = model(**tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(device))[0]
    probs = probs[0].detach().cpu().numpy()
    label = probs.argmax().item()
    explainer = lime_text.LimeTextExplainer(kernel_width=kernel_width, verbose=False, random_state=random_seed)
    model_regressor = Ridge(alpha=alpha, fit_intercept=True,
                                    random_state=random_seed)
    
    def batch_predict(texts):
        model.eval()
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = batch.to(device)

        logits = model(**batch)

        return logits[0].detach().cpu().numpy()
    
    explanation = explainer.explain_instance(x, 
                                         batch_predict, # classification function
                                         distance_metric=distance_metric,
                                         labels=(label,),
                                         num_samples=n_samples,
                                         batch_size=128,
                                         model_regressor=model_regressor,
                                         distribution=distribution)
    

    return {'local_exp':explanation.local_exp[label], 
            'intercept':explanation.intercept[label],
            'score':explanation.score[label],
            'local_pred':explanation.local_pred[label],
            'true_prob':probs[label]}

def lime_tabular_explanation(model, x, x_cols, train_dataset, device, random_seed=2023, n_samples=100,
                     kernel_width=3, distribution='exp', weighted=False,
                     distance_metric='cosine',
                     alpha=1):
    set_seed(random_seed)
    probs = model.predict(pd.DataFrame([x], columns=x_cols)).iloc[:,-3:-1].values[0]
    label = probs.argmax()
    explainer = lime_tabular.LimeTabularExplainer(training_data=train_dataset, kernel_width=kernel_width, verbose=False, random_state=random_seed)
    model_regressor = Ridge(alpha=alpha, fit_intercept=True,
                                    random_state=random_seed)
    
    def batch_predict(batch):
        batch = pd.DataFrame(batch, columns=x_cols)
        probs = model.predict(batch).iloc[:,-3:-1].values
        return probs
    
    explanation = explainer.explain_instance(x, 
                                         batch_predict, 
                                         distance_metric=distance_metric,
                                         labels=(label,),
                                         num_samples=n_samples,
                                         batch_size=128,
                                         model_regressor=model_regressor,
                                         distribution=distribution)
    

    return {'local_exp':explanation.local_exp[label], 
            'intercept':explanation.intercept[label],
            'score':explanation.score[label],
            'local_pred':explanation.local_pred[label],
            'true_prob':probs[label]}


def jaccard_index(a, b):
    a = set(a)
    b = set(b)
    return len(a & b)/ len(a | b)


def set_seed(seed=None):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def calculate_metrics(attributions):
    r2 = np.mean([xx['score'] for xx in attributions])
    residual = np.mean([np.abs(xx['local_pred'] - xx['true_prob']) for xx in attributions])
    
    n_features = len(attributions[0]['local_exp'])
    jaccard_indexes = np.zeros(n_features)
    for k in range(1, n_features + 1):
        for i in range(len(attributions) - 1):
            attr_i = attributions[i]['local_exp']
            attr_i = sorted(attr_i, key=lambda xx:xx[1])
            attr_i = [xx[0] for xx in attr_i]
            for j in range(i + 1, len(attributions)):
                attr_j = attributions[j]['local_exp']
                attr_j = sorted(attr_j, key=lambda xx:xx[1])
                attr_j = [xx[0] for xx in attr_j]
                
                jaccard_indexes[k-1] += jaccard_index(attr_i[-k:], attr_j[-k:])
        
    jaccard_indexes /= (len(attributions)*(len(attributions)-1)/2)
    
    ji = np.mean(jaccard_indexes)
    return {'R2':r2,
            'JaccardIndex':ji,
            'Residual':residual}
