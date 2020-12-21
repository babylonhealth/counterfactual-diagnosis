# Counterfactional Diagnosis

This codebase and accompanying files is intended for the purposes of replicating the experimental results reported in [1], and for enabling researchers to try our counterfactual diagnostic algorithms on their own models. The code and accompanying files are not to be shared, published or used in any other manner without the express authorisation of the authors of [1]. Those requesting these files agree to not disclose their contents to third parties. The inference engine used in this code release is not the propreitary inference engine used in production by Babylon Partners Limited, and is designed to run on a personal computer. As a result, some probabilistic results may vary to the order of 0.1% due to numerical differences. The code and data included in the release was generated for the purposes of this study, and does not reflect any model or implimentation in production currently or at any previous point at Babylon Partners Limited.

The next release of the code will include an API to allow the vignettes to be run on the models used in the [1]. In the interim, the --reproduce flag runs the code on precompied results to produce the results presented in [1]. Running run.py without --reproduce will run the inference engine on the models included in the repo, which are random noisy-OR networks. To run your own models, replace these and run the create_marginals method. 

For any further enquiries, please contact the lead author at jonathan.richens@babylonhealth.com, I'd be happy to help!

[1] Richens, Jonathan G., Ciarán M. Lee, and Saurabh Johri. "Improving the accuracy of medical diagnosis with causal machine learning." Nature communications 11.1 (2020): 1-9.

## Requirements
Use `python > 3.6` and install the requirements:

```
pip install -r requiurements.txt
```

## Running

To reproduce the results from the paper:

```
python run.py --results my_results_dir --reproduce
```

To run on your own models, replace data/test_networks with your own network dictionary. E.g.

```
python run.py --first 10 
```
