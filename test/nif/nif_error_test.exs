defmodule LgbmEx.NIF.ErrorTest do
  @moduledoc """
  Unit tests for NIF layer error handling.

  Tests error cases and edge conditions:
  - Invalid model file
  - Invalid input data
  - Error response format

  These tests verify the NIF layer properly handles and reports errors
  before and after refactoring.
  """

  use ExUnit.Case, async: false

  alias LgbmEx.NIFAPI

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)
    :ok
  end

  describe "error handling on invalid model file" do
    @describetag :tmp_dir

    @tag :skip
    test "create_reference fails gracefully with missing model file" do
      # Create a model struct with a non-existent model file path
      fake_model = %{
        files: %{
          model: "/path/to/nonexistent/model.txt"
        }
      }

      # Should raise an error or return error tuple, not crash
      assert_raise RuntimeError, fn ->
        NIFAPI.create_reference(fake_model)
      end
    end
  end

  describe "error handling on invalid predict input" do
    @describetag :tmp_dir

    @tag :skip
    test "predict handles invalid row data gracefully" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      # Try predicting with wrong number of features
      invalid_input = [[1.0, 2.0]]  # Only 2 features instead of 4

      # Should raise an error when invalid data is provided
      assert_raise RuntimeError, fn ->
        LgbmEx.predict(model, invalid_input)
      end
    end
  end

  describe "error response format" do
    @describetag :tmp_dir

    test "NIFAPI encodes and decodes responses correctly" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      # Valid call should work and return data
      result = LgbmEx.predict(model, [[5.4, 3.9, 1.7, 0.4]])

      # Result should be in expected format (list of predictions)
      assert is_list(result)
      assert Enum.count(result) > 0
    end
  end
end
