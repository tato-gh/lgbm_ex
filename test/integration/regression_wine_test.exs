defmodule LgbmEx.Integration.RegressionWineTest do
  @moduledoc """
  Integration tests for regression using Wine dataset.

  Tests the regression pipeline with real data:
  - Raw list input (single/multiple rows)
  - DataFrame input
  - Regression output validation with domain knowledge

  These tests focus on end-to-end functionality with actual dataset,
  not edge cases or low-level API behavior.
  """

  use ExUnit.Case, async: false

  alias Explorer.DataFrame, as: DF

  # Sample wine features (excluding class and alcohol which is target)
  # [class, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids,
  #  nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline]
  @sample_wine_1 [1, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
  @sample_wine_2 [1, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]
  @sample_wine_3 [1, 2.36, 2.67, 18.6, 101, 2.8, 3.24, 0.3, 2.81, 5.68, 1.03, 3.17, 1185]

  # Typical alcohol content range for wine: 11-16%
  @min_alcohol 10.0
  @max_alcohol 16.0

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)

    df = Explorer.Datasets.wine()

    model =
      LgbmEx.fit_without_val("regression_test", df, "alcohol",
        objective: "regression",
        metric: "rmse",
        num_iterations: 5
      )

    {:ok, model: model, df: df}
  end

  describe "regression with raw list input" do
    @describetag :tmp_dir

    test "single row prediction", %{model: model} do
      x_test = [@sample_wine_1]
      [[prediction]] = LgbmEx.predict(model, x_test)

      assert is_number(prediction)
      assert prediction >= @min_alcohol and prediction <= @max_alcohol
    end

    test "multiple rows prediction", %{model: model} do
      x_test = [@sample_wine_1, @sample_wine_2, @sample_wine_3]

      [[p1], [p2], [p3]] = LgbmEx.predict(model, x_test)

      for prediction <- [p1, p2, p3] do
        assert is_number(prediction)
        assert prediction >= @min_alcohol and prediction <= @max_alcohol
      end
    end
  end

  describe "regression with DataFrame input" do
    @describetag :tmp_dir

    test "small batch from DataFrame", %{model: model, df: df} do
      # Take first 3 rows and exclude alcohol (target variable)
      x_test = DF.slice(df, 0, 3) |> DF.discard("alcohol")

      [[p1], [p2], [p3]] = LgbmEx.predict(model, x_test)

      for prediction <- [p1, p2, p3] do
        assert is_number(prediction)
        assert prediction >= @min_alcohol and prediction <= @max_alcohol
      end
    end

    test "larger batch from DataFrame", %{model: model, df: df} do
      x_test = DF.slice(df, 0, 20) |> DF.discard("alcohol")

      result = LgbmEx.predict(model, x_test)

      assert Enum.count(result) == 20

      for [prediction] <- result do
        assert is_number(prediction)
        assert prediction >= @min_alcohol and prediction <= @max_alcohol
      end
    end
  end

  describe "training with validation data" do
    @describetag :tmp_dir

    test "model trained with separate validation data predicts correctly" do
      df = Explorer.Datasets.wine()

      # Split into train (80%) and validation (20%)
      total_rows = DF.n_rows(df)
      train_size = div(total_rows * 4, 5)

      df_train = DF.slice(df, 0, train_size)
      df_val = DF.slice(df, train_size, total_rows - train_size)

      model =
        LgbmEx.fit("regression_with_val", {df_train, df_val}, "alcohol",
          objective: "regression",
          metric: "rmse",
          num_iterations: 10
        )

      # Verify model can predict
      x_test = [@sample_wine_1, @sample_wine_2, @sample_wine_3]
      [[p1], [p2], [p3]] = LgbmEx.predict(model, x_test)

      for prediction <- [p1, p2, p3] do
        assert is_number(prediction)
        assert prediction >= @min_alcohol and prediction <= @max_alcohol
      end
    end

    test "model with validation has evaluation metrics" do
      df = Explorer.Datasets.wine()

      # Split into train and validation
      total_rows = DF.n_rows(df)
      train_size = div(total_rows * 4, 5)

      df_train = DF.slice(df, 0, train_size)
      df_val = DF.slice(df, train_size, total_rows - train_size)

      model =
        LgbmEx.fit("regression_with_eval", {df_train, df_val}, "alcohol",
          objective: "regression",
          metric: "rmse",
          num_iterations: 10
        )

      # Get eval results
      eval_result = LgbmEx.NIF.booster_get_eval(model.ref)

      # Should return non-empty charlist when validation data exists
      assert is_list(eval_result)
      assert eval_result != []
    end
  end

  describe "regression output validation with domain knowledge" do
    @describetag :tmp_dir

    test "predictions are within realistic alcohol percentage range", %{model: model} do
      x_test = [@sample_wine_1, @sample_wine_2, @sample_wine_3]

      [[p1], [p2], [p3]] = LgbmEx.predict(model, x_test)

      # All predictions should be scalar numbers
      for prediction <- [p1, p2, p3] do
        assert is_number(prediction)
        # Wine alcohol content is typically 11-16%, allowing some margin
        assert prediction >= @min_alcohol and prediction <= @max_alcohol
      end
    end

    test "different wine characteristics produce different alcohol predictions", %{model: model} do
      # Different wine samples should have different alcohol predictions
      x_test = [@sample_wine_1, @sample_wine_2]

      [[p1], [p2]] = LgbmEx.predict(model, x_test)

      # Both should be valid predictions
      assert is_number(p1)
      assert is_number(p2)
      assert p1 >= @min_alcohol and p1 <= @max_alcohol
      assert p2 >= @min_alcohol and p2 <= @max_alcohol

      # Different samples should generally produce different predictions
      assert p1 != p2
    end
  end
end
