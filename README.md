# GLIME-General-Stable-and-Local-LIME-Explanation
GLIME is a post-hoc explanation method that solves the instability problem arised in LIME. GLIME unifies several previous methods including LIME, KernelSHAP, SmoothGrad, Gradient, DLIME, ALIME. 

[GLIME: General, Stable and Local LIME Explanation](https://openreview.net/forum?id=3FJaFElIVN&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions)) (NeurIPS 2023 Spotlight)

The code is adopted from the implementation of [LIME](https://github.com/marcotcr/lime)

## How to use
GLIME can be used in the same way as LIME. 

```python
# x is the input image, model is the black-box model to be explained. 
kernel_width = 0.25
random_seed=2023
label = 1
n_samples = 1024
distribution = 'uniform'
segmenter = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                        max_dist=200, ratio=0.2,
                                                        random_seed=2023)

explainer = lime_image.LimeImageExplainer(kernel_width=kernel_width,verbose=False, random_state=random_seed)

model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=random_seed)


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(get_preprocess_transform()(i) for i in images), dim=0)

    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


explanation = explainer.explain_instance(x, 
                                        batch_predict, # classification function
                                        segmentation_fn=segmenter,
                                        hide_color=0, 
                                        distance_metric='l2',
                                        top_labels=None,
                                        labels=(label,),
                                        num_samples=n_samples,
                                        batch_size=128,
                                        model_regressor=model_regressor,
                                        distribution=distribution,
                                        random_seed=random_seed,
                                        model=model)
```

`distribution=uniform` is the distribution setting for LIME. To use GLIME-Binomial, one can set `distribution=comb_exp`. 

Users can also define their own sampling distributions in `utils/generic_utils.py`.
