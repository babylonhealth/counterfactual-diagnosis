# Improving the accuracy of medical diagnosis with causal machine learning

This is to accompany the paper [1], and is intended for the purposes of i) replicating the experimental results reported in [1], and ii) enabling researchers to try our counterfactual diagnostic algorithms on their own models. The code and accompanying files are not to be published or used for any other purpose without the express authorisation of the authors of [1]. The counterfactual ranking algorithm currently under patent applucation [2]. The inference engine used in this code is designed to run on a personal computer, and as a result, some probabilistic results may vary to the order of 0.1% due to numerical differences. The code, models and data included in the release was generated for the purposes of this study, and does not reflect any model or implimentation in production currently or at any previous point at Babylon Partners Limited.

For any further enquiries, please contact the lead author at jonathan.richens@babylonhealth.com, I'd be happy to help!

[1] Richens, Jonathan G., Ciarán M. Lee, and Saurabh Johri. "Improving the accuracy of medical diagnosis with causal machine learning." Nature communications 11.1 (2020): 1-9.

[2] https://patents.google.com/patent/US20200279655A1/en

## Paper

Improving the accuracy of medical diagnosis with causal machine learning, by Jonathan G. Richens, Ciarán M. Lee & Saurabh Johri. Published in nature communications. 

URL https://www.nature.com/articles/s41467-020-17419-7

## Citation
```
@article{richens2020improving,
  title={Improving the accuracy of medical diagnosis with causal machine learning},
  author={Richens, Jonathan G and Lee, Ciar{\'a}n M and Johri, Saurabh},
  journal={Nature communications},
  volume={11},
  number={1},
  pages={1--9},
  year={2020},
  publisher={Nature Publishing Group}
}
```

## Requirements

Use `python > 3.6` and install the requirements:

```
pip install -r requiurements.txt
```

## Running

The upcoming release of the code will include an API to allow the vignettes to be run on the models used in the [1]. In the interim, the --reproduce flag runs the code on precompied results to produce the results presented in [1]. Running run.py without --reproduce will run the inference engine on the models included in the repo, which are random noisy-OR networks. To run your own models, replace these and run the create_marginals method. 

**Please note that any information not used to generate the results in [1] is not included in the clinical vignettes. This includes doctor and patient information, and specifics such as disease and symptom concepts (other than the rareness of a disease)**

To reproduce the results from the paper:

```
python run.py --results my_results_dir --reproduce
```

To run on your own models, replace data/test_networks with your own network dictionary.

```
python run.py --first 10 
```

## Copyright and Licence

Copyright 2019-2020 Babylon Health (Babylon Partners Limited).

GNU General Public License v3.0

The algorithms presented in [1] and implimented here are protected under U.S. Patent Application No. 16/520,280. 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
