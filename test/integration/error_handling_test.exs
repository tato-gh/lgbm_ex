defmodule LgbmEx.Integration.ErrorHandlingTest do
  @moduledoc """
  Integration tests for error handling in realistic scenarios.

  Tests error cases with real datasets to ensure the system
  gracefully handles invalid inputs and model issues.

  Note: Many error conditions in LightGBM C++ library cause fatal errors
  (segfaults) that cannot be caught by Elixir exception handling.
  These tests focus on errors that can be caught at the Elixir level.
  """

  use ExUnit.Case, async: false

  alias Explorer.DataFrame, as: DF

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)

    # Create a valid model for testing
    {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

    model =
      LgbmEx.fit_without_val("error_test", df, "species",
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 10,
        learning_rate: 0.1
      )

    {:ok, model: model}
  end

  describe "DataFrame input validation" do
    @describetag :tmp_dir

    test "DataFrame with missing required columns raises error", %{model: model} do
      # Create DataFrame with only 2 columns (missing petal_length and petal_width)
      invalid_df =
        DF.new(%{
          sepal_length: [5.1, 4.9],
          sepal_width: [3.5, 3.0]
        })

      # Should raise ArgumentError because required columns are missing
      assert_raise ArgumentError, fn ->
        LgbmEx.predict(model, invalid_df)
      end
    end
  end
end
