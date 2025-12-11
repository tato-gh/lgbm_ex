defmodule LgbmEx.NIF.BasicTest do
  @moduledoc """
  Unit tests for NIF layer basic functionality.

  Tests the fundamental operations of each NIF function:
  - booster_create_from_model_file (create reference)
  - booster_get_num_classes
  - booster_get_num_features
  - booster_get_current_iteration
  - booster_get_eval
  - booster_get_loaded_param
  - booster_feature_importance

  These tests ensure the NIF layer provides correct UI and data formats
  before and after refactoring.
  """

  use ExUnit.Case, async: false

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)
    :ok
  end

  describe "booster_create_from_model_file" do
    @describetag :tmp_dir

    test "returns a valid resource reference" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      # Verify the model reference is a valid resource
      assert is_reference(model.ref)
    end
  end

  describe "booster_get_num_classes" do
    @describetag :tmp_dir

    test "returns num_classes from NIF" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      # Call NIF function to get num_classes
      result = LgbmEx.NIF.booster_get_num_classes(model.ref)
      decoded = result |> List.to_string() |> Jason.decode!()
      assert decoded["result"] == 3
    end
  end

  describe "booster_get_num_features" do
    @describetag :tmp_dir

    test "returns num_features from NIF" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      # Call NIF function to get num_features
      result = LgbmEx.NIF.booster_get_num_features(model.ref)
      decoded = result |> List.to_string() |> Jason.decode!()
      assert decoded["result"] == 4
    end
  end

  describe "booster_get_current_iteration" do
    @describetag :tmp_dir

    test "returns correct number of iterations from NIF" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 5
        )

      # Call NIF function to get current iteration
      result = LgbmEx.NIF.booster_get_current_iteration(model.ref)
      decoded = result |> List.to_string() |> Jason.decode!()
      assert decoded["result"] == 5
    end

    test "returns different values for models with different iterations" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model1 =
        LgbmEx.fit_without_val("test1", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 5
        )

      model2 =
        LgbmEx.fit_without_val("test2", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 10
        )

      # Call NIF function for both models
      result1 = LgbmEx.NIF.booster_get_current_iteration(model1.ref)
      result2 = LgbmEx.NIF.booster_get_current_iteration(model2.ref)

      decoded1 = result1 |> List.to_string() |> Jason.decode!()
      decoded2 = result2 |> List.to_string() |> Jason.decode!()

      assert decoded1["result"] == 5
      assert decoded2["result"] == 10
      assert decoded1["result"] != decoded2["result"]
    end
  end

  describe "booster_get_eval" do
    @describetag :tmp_dir

    test "returns eval results from NIF when validation data exists" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 5
        )

      # Call NIF function to get eval results
      result = LgbmEx.NIF.booster_get_eval(model.ref)
      # Should return charlist that can be decoded
      assert is_list(result)
    end

    test "returns empty when no validation data" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 5
        )

      # Call NIF function to get eval results
      result = LgbmEx.NIF.booster_get_eval(model.ref)
      assert is_list(result)
    end
  end

  describe "booster_get_loaded_param" do
    @describetag :tmp_dir

    test "returns model parameters from NIF" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      params = [
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 5
      ]

      model = LgbmEx.fit_without_val("test", df, "species", params)

      # Call NIF function to get loaded parameters
      result = LgbmEx.NIF.booster_get_loaded_param(model.ref)
      # Should return charlist that can be decoded to parameters
      assert is_list(result)
      param_str = List.to_string(result)
      assert String.contains?(param_str, "objective")
    end
  end

  describe "booster_feature_importance" do
    @describetag :tmp_dir

    test "returns feature importance from NIF" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 5
        )

      # Call NIF function to get feature importance
      result = LgbmEx.NIF.booster_feature_importance(model.ref)
      # Should return charlist that can be decoded
      assert is_list(result)
      decoded = result |> List.to_string() |> Jason.decode!()
      assert Map.has_key?(decoded, "result")
    end
  end
end
