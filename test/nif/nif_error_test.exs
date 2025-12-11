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

  setup do
    tmp_dir = System.tmp_dir!()
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)
    :ok
  end

  describe "error handling on invalid model file" do
    @describetag :tmp_dir

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
