# LgbmEx

LgbmEx is a wrapper library for microsoft/LightGBM (partical) cli, C-API implemented by Elixir.

**NOTE**

- Beta version / Not stable
- The building of the model uses CLI commands.
  - This means using a disk for model and data storage.
- Prediction uses the C-API.


## Sample

```elixir

# Please set environment LGBM_EX_WORKDIR for the working disk space, otherwise, uses `System.tmp_dir()/lgbm_ex`

{_mapping, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

model =
  LgbmEx.fit("model_directory_name", df, "species",
    objective: "multiclass",
    metric: "multi_logloss",
    num_class: 3,
    num_iterations: 20
  )

x_test =
 [
   [5.4, 3.9, 1.7, 0.4],
   [5.7, 2.8, 4.5, 1.4],
   [7.6, 3.0, 6.6, 2.2]
 ]

LgbmEx.predict(model, x_test)
```


## Instration

```elixir
def deps do
  [
    {:lgbm_ex, "0.0.2", github: "/tato-gh/lgbm_ex"}
  ]
end
```

- Probably does not work on os x. Docker recommended.


## Refereance

- [Welcome to LightGBM’s documentation!](https://lightgbm.readthedocs.io/en/latest/)


## License

Copyright 2024 tato

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

