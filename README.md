# Clarifying Questions

This is the repository for our paper [Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions](https://arxiv.org/abs/2410.13788).

For any additional questions, please feel free to contact me. My email can be found in our paper, linked above.

## Data
Our datasets used for training with human-identified answer sets are in `data/ambigqa.train_4k.clarify.jsonl` and `data/ambigqa.dev_4h.clarify.jsonl`.

Likewise, our datasets with model-identified answer sets can be found in `data/nqopen.train_4k.clarify.jsonl` and `data/nqopen.dev_4h.clarify.jsonl`.

## Training & Inference
The primary scripts used for training can be found in for both stages of training are in `src/sft.py` and `src/dpo.py`.

The `src/inference.py` script contains code for performing all stages of inference: generating direct-answers, clarifying questions, clarifying answers, and answers-with-clarification.

Generations can be be labeled for (simulated) preference with `src/clarifying_rewards.py` and evaluated with `src/metrics.py`.

## Citations
If you find our work helpful, please cite us:

```
@article{zhang2025modeling,
  title={Modeling future conversation turns to teach llms to ask clarifying questions},
  author={ Zhang, Michael J.Q. and Knox, W. Bradley and Choi, Eunsol },
  journal={ International Conference on Learning Representations (ICLR) },
  year={ 2025 }
}
```

If you use any of the data from this work, please cite the curators of the original datasets:
```
@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics (TACL) },
  year={ 2019 }
}

@inproceedings{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    journal={ Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) },
    year={ 2020 }
}
```
