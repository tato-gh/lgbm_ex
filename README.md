# LgbmEx

LgbmEx is a wrapper library for microsoft/LightGBM (partical) cli, C-API implemented by Elixir.

**NOTE**

- Beta version / Not stable
- Building model uses CLI command
- Prediction uses C-API


## Sample

```elixir

features_train = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [4.6, 3.1, 1.5, 0.2],
  [5.0, 3.6, 1.4, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5],
  [6.9, 3.1, 4.9, 1.5],
  [5.5, 2.3, 4.0, 1.3],
  [6.5, 2.8, 4.6, 1.5],
  [6.3, 3.3, 6.0, 2.5],
  [5.8, 2.7, 5.1, 1.9],
  [7.1, 3.0, 5.9, 2.1],
  [6.3, 2.9, 5.6, 1.8],
  [6.5, 3.0, 5.8, 2.2]
]

labels_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

features = [
  [5.4, 3.9, 1.7, 0.4],
  [5.7, 2.8, 4.5, 1.3],
  [7.6, 3.0, 6.6, 2.1]
]

tmp_dir = System.tmp_dir
model = Lgbm_ex.new_model(tmp_dir)
model = LgbmEx.fit(features_train, labels_train, [
  objective: "multiclass",
  metric: "multi_logloss",
  num_class: 3,
  num_iterations: 10,
  num_leaves: 5,
  min_data_in_leaf: 1
])

LgbmEx.predict(model, features)
```


## Instration

```elixir
def deps do
  [
    {:lgbm_ex, "0.0.1", github: "/tato-gh/lgbm_ex"}
  ]
end
```

- Probably does not work on os x. Docker recommended.


## Refereance

- [Welcome to LightGBM’s documentation!](https://lightgbm.readthedocs.io/en/latest/)


## License

Copyright 2024 ta.to.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

