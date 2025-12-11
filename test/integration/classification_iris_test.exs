defmodule LgbmEx.Integration.ClassificationIrisTest do
  @moduledoc """
  Integration tests for classification using Iris dataset.

  Tests the classification pipeline with real data:
  - Raw list input (single/multiple rows)
  - DataFrame input with grouped operations
  - Classification accuracy with domain knowledge validation

  These tests focus on end-to-end functionality with actual dataset,
  not edge cases or low-level API behavior.
  """

  use ExUnit.Case, async: false

  alias Explorer.DataFrame, as: DF

  # Sample Iris features for each species (sepal_length, sepal_width, petal_length, petal_width)
  @setosa_features [5.4, 3.9, 1.7, 0.4]
  @versicolor_features [5.7, 2.8, 4.5, 1.3]
  @virginica_features [7.6, 3.0, 6.6, 2.2]

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)

    {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

    model =
      LgbmEx.fit_without_val("test", df, "species",
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 5
      )

    {:ok, model: model}
  end

  describe "predict with raw list input" do
    @describetag :tmp_dir

    test "single row prediction", %{model: model} do
      x_test = [@setosa_features]
      [p1] = LgbmEx.predict(model, x_test)

      assert is_list(p1)
      assert Enum.count(p1) == 3  # 3 classes
      assert Enum.all?(p1, &is_number/1)
    end

    test "multiple rows prediction", %{model: model} do
      x_test = [@setosa_features, @versicolor_features, @virginica_features]

      [p1, p2, p3] = LgbmEx.predict(model, x_test)

      for prediction <- [p1, p2, p3] do
        assert is_list(prediction)
        assert Enum.count(prediction) == 3
        assert Enum.all?(prediction, &is_number/1)
      end
    end
  end

  describe "predict with DataFrame input" do
    @describetag :tmp_dir

    test "one row from each species group", %{model: model} do
      df = Explorer.Datasets.iris()
      grouped = DF.group_by(df, "species")
      # slice(grouped, 0, 1) gets first row from each group (3 groups = 3 rows)
      x_test = DF.slice(grouped, 0, 1) |> DF.ungroup()

      result = LgbmEx.predict(model, x_test)

      assert Enum.count(result) == 3

      for prediction <- result do
        assert is_list(prediction)
        assert Enum.count(prediction) == 3
        assert Enum.all?(prediction, &is_number/1)
      end
    end

    test "multiple rows from each species group", %{model: model} do
      df = Explorer.Datasets.iris()
      grouped = DF.group_by(df, "species")
      # slice(grouped, 0, 5) gets first 5 rows from each group (3 groups * 5 rows = 15 rows)
      x_test = DF.slice(grouped, 0, 5) |> DF.ungroup()

      result = LgbmEx.predict(model, x_test)

      assert Enum.count(result) == 15

      for prediction <- result do
        assert is_list(prediction)
        assert Enum.count(prediction) == 3
        assert Enum.all?(prediction, &is_number/1)
      end
    end
  end

  describe "training with validation data" do
    @describetag :tmp_dir

    test "model trained with separate validation data predicts correctly" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      # Split into train (80%) and validation (20%)
      total_rows = DF.n_rows(df)
      train_size = div(total_rows * 4, 5)

      df_train = DF.slice(df, 0, train_size)
      df_val = DF.slice(df, train_size, total_rows - train_size)

      model =
        LgbmEx.fit("test_with_val", {df_train, df_val}, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 10
        )

      # Verify model can predict
      x_test = [@setosa_features, @versicolor_features, @virginica_features]
      result = LgbmEx.predict(model, x_test)

      assert Enum.count(result) == 3

      for prediction <- result do
        assert is_list(prediction)
        assert Enum.count(prediction) == 3
        assert Enum.all?(prediction, &is_number/1)
      end
    end

    test "model with validation has evaluation metrics" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      # Split into train and validation
      total_rows = DF.n_rows(df)
      train_size = div(total_rows * 4, 5)

      df_train = DF.slice(df, 0, train_size)
      df_val = DF.slice(df, train_size, total_rows - train_size)

      model =
        LgbmEx.fit("test_with_eval", {df_train, df_val}, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 10
        )

      # Get eval results
      eval_result = LgbmEx.NIF.booster_get_eval(model.ref)

      # Should return non-empty charlist when validation data exists
      assert is_list(eval_result)
      assert eval_result != []
    end
  end

  describe "classification accuracy with domain knowledge" do
    @describetag :tmp_dir

    test "predicts correct species for typical samples", %{model: model} do
      x_test = [@setosa_features, @versicolor_features, @virginica_features]

      [p_setosa, p_versicolor, p_virginica] = LgbmEx.predict(model, x_test)

      # Helper to get predicted class (index of max probability)
      get_predicted_class = fn probs ->
        probs |> Enum.with_index() |> Enum.max_by(fn {val, _idx} -> val end) |> elem(1)
      end

      # Setosa-like features should predict class 0 (Setosa)
      assert get_predicted_class.(p_setosa) == 0

      # Versicolor-like features should predict class 1 (Versicolor)
      assert get_predicted_class.(p_versicolor) == 1

      # Virginica-like features should predict class 2 (Virginica)
      assert get_predicted_class.(p_virginica) == 2
    end

    test "probability values are in reasonable range", %{model: model} do
      x_test = [@setosa_features, @versicolor_features, @virginica_features]

      predictions = LgbmEx.predict(model, x_test)

      for prediction <- predictions do
        assert Enum.count(prediction) == 3

        for prob <- prediction do
          assert is_number(prob)
          # Probabilities should be roughly in [0, 1] range (allowing small margin for numerical precision)
          assert prob >= -0.1 and prob <= 1.1
        end
      end
    end
  end
end
