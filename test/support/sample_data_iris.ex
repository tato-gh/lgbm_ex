defmodule LgbmEx.SampleDataIris do
  @moduledoc """
  Sample data for test.

  Clipping of iris dataset.
  """

  def train_set do
    {
      [
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
      ],
      [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2
      ]
    }
  end

  def test_set do
    {
      [
        [5.4, 3.9, 1.7, 0.4],
        [5.4, 3.9, 1.7, 0.5],
        [5.7, 2.8, 4.5, 1.3],
        [5.7, 2.8, 4.5, 1.4],
        [7.6, 3.0, 6.6, 2.1],
        [7.6, 3.0, 6.6, 2.2]
      ],
      [0, 0, 1, 1, 2, 2]
    }
  end

  def parameters do
    [
      objective: "multiclass",
      metric: "multi_logloss",
      num_class: 3,
      num_iterations: 10,
      num_leaves: 5,
      min_data_in_leaf: 1,
      min_gain_to_split: 1,
      force_row_wise: true,
      seed: 42,
      verbose: 1
    ]
  end

  def parameters_with_early_stopping do
    parameters()
    |> Keyword.merge([
      num_iterations: 100,
      early_stopping_round: 3
    ])
  end
end
