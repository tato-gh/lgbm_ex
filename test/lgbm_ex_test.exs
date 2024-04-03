defmodule LgbmExTest do
  use ExUnit.Case, async: false

  alias Explorer.DataFrame, as: DF

  setup(%{tmp_dir: tmp_dir}) do
    Application.put_env(:lgbm_ex, :workdir, tmp_dir)
    :ok
  end

  describe "fit" do
    @describetag :tmp_dir

    test "returns model" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 20
        )

      assert model.num_iterations == 20
      assert Enum.count(model.learning_steps) == 20
      assert Enum.count(Keyword.get(model.parameters, :x_names)) == 4
      assert Enum.count(model.feature_importance_split) == 4
      assert Enum.count(model.feature_importance_gain) == 4
      assert Keyword.get(model.parameters, :y_name) == "species"
    end

    test "early_stopping" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      grouped = DF.group_by(df, "species")
      df_train = DF.slice(grouped, 0, 20) |> DF.ungroup()
      df_val = DF.slice(grouped, 20, 5) |> DF.ungroup()

      model =
        LgbmEx.fit("test", {df_train, df_val}, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 100,
          early_stopping_round: 1,
          learning_rate: 0.3
        )

      assert model.num_iterations < 100
    end
  end

  describe "fit_without_val" do
    @describetag :tmp_dir

    test "returns model" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 20
        )

      assert Enum.count(model.learning_steps) == 0
    end
  end

  describe "predict" do
    @describetag :tmp_dir

    setup do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      LgbmEx.fit_without_val("test", df, "species",
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 20
      )

      :ok
    end

    test "returns predicted values, case raw list data given" do
      model = LgbmEx.load_model("test")

      x_test =
        [
          [5.4, 3.9, 1.7, 0.4],
          [5.7, 2.8, 4.5, 1.4],
          [7.6, 3.0, 6.6, 2.2]
        ]

      [p1, p2, p3] = LgbmEx.predict(model, x_test)

      assert Enum.at(p1, 0) > 0.5
      assert Enum.at(p2, 1) > 0.5
      assert Enum.at(p3, 2) > 0.5
    end

    test "returns predicted values, case %DataFrame{} given" do
      model = LgbmEx.load_model("test")
      df = Explorer.Datasets.iris()
      grouped = DF.group_by(df, "species")
      x_test = DF.slice(grouped, 0, 1) |> DF.ungroup()

      [p1, p2, p3] = LgbmEx.predict(model, x_test)

      assert Enum.at(p1, 0) > 0.5
      assert Enum.at(p2, 1) > 0.5
      assert Enum.at(p3, 2) > 0.5
    end
  end

  describe "refit_model" do
    @describetag :tmp_dir

    test "returns result by given new parameters" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 20
        )

      assert 20 == Enum.count(model.learning_steps)

      model = LgbmEx.refit_model(model, num_iterations: 2)
      assert 2 == Enum.count(model.learning_steps)
    end
  end

  describe "load_model" do
    @describetag :tmp_dir

    test "returns model loaded" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      LgbmEx.fit_without_val("test", df, "species",
        objective: "multiclass",
        metric: "multi_logloss",
        num_class: 3,
        num_iterations: 2
      )

      assert LgbmEx.load_model("test")
    end
  end

  describe "copy_model" do
    @describetag :tmp_dir

    test "returns model copied" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit_without_val("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 2
        )

      copied_model = LgbmEx.copy_model(model, "copied")
      assert File.exists?(copied_model.files.model)
    end
  end

  describe "Persist" do
    @describetag :tmp_dir

    test "zip and unzip" do
      {_, df} = Explorer.Datasets.iris() |> LgbmEx.preproccessing_label_encode("species")

      model =
        LgbmEx.fit("test", df, "species",
          objective: "multiclass",
          metric: "multi_logloss",
          num_class: 3,
          num_iterations: 20
        )

      zip_path = LgbmEx.zip_model(model)

      assert File.exists?(zip_path)
      assert String.ends_with?(zip_path, "/test.zip")

      model = LgbmEx.unzip_model(zip_path, "test_from_zip")

      x_test = [[5.4, 3.9, 1.7, 0.4]]
      [p1] = LgbmEx.predict(model, x_test)
      assert Enum.at(p1, 0) > 0.5
    end
  end
end
