defmodule LgbmExTest do
  use ExUnit.Case, async: false
  doctest LgbmEx

  alias LgbmEx.SampleDataIris

  def setup_iris_model(%{tmp_dir: tmp_dir}) do
    {x, y} = SampleDataIris.train_set()
    parameters = SampleDataIris.parameters()
    model = LgbmEx.new_model(tmp_dir)
    model = LgbmEx.fit(model, x, y, parameters)
    LgbmEx.save_as(model, "iris")
    :ok
  end

  describe "fit" do
    @describetag :tmp_dir

    test "returns model", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, x, y, parameters)

      assert model.num_iterations == 10
      assert Enum.count(model.learning_steps) == 10
    end

    test "early stopping and returns steps", %{
      tmp_dir: tmp_dir
    } do
      {x_train, y_train} = SampleDataIris.train_set()
      {x_val, y_val} = SampleDataIris.test_set()
      parameters = SampleDataIris.parameters_with_early_stopping()

      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, {x_train, x_val}, {y_train, y_val}, parameters)

      assert model.num_iterations > 10
      assert Enum.count(model.learning_steps) > 10
      assert hd(model.learning_steps) == {0, 0.939663}
    end
  end

  describe "fit_without_eval" do
    @describetag :tmp_dir

    test "returns model", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit_without_eval(model, x, y, parameters)

      assert model.num_iterations == 10
      # cannot get values because of unuse early_stopping
      assert model.learning_steps == []
    end
  end

  describe "refit" do
    @describetag :tmp_dir

    test "returns result by given new parameters", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, x, y, parameters)

      model = LgbmEx.refit(model, num_iterations: 2)
      assert model.num_iterations == 2
    end
  end

  describe "grid_search" do
    @describetag :tmp_dir

    test "returns results by given grid parameters", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, x, y, parameters)

      grid = [
        num_iterations: [5, 10],
        min_data_in_leaf: [2, 3]
      ]

      models = LgbmEx.grid_search(model, grid)

      assert 4 == Enum.count(models)

      assert [{5, 2}, {10, 2}, {5, 3}, {10, 3}] ==
        Enum.map(models, & {
          Keyword.get(&1.parameters, :num_iterations),
          Keyword.get(&1.parameters, :min_data_in_leaf)
        })
    end
  end

  describe "cross_validate" do
    @describetag :tmp_dir

    setup %{tmp_dir: tmp_dir} do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()
      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, x, y, parameters)

      %{model: model}
    end

    test "returns cross_validated results", %{
      model: model
    } do
      {x_test, _y} = SampleDataIris.test_set()
      results = LgbmEx.cross_validate(model, x_test, 3)

      assert 3 == Enum.count(results)
    end

    test "cross_validation with shuffle", %{
      model: model
    } do
      {x_test, _y} = SampleDataIris.test_set()
      results = LgbmEx.cross_validate(model, x_test, 3, :shuffle)

      assert 3 == Enum.count(results)
    end
  end

  describe "predict" do
    @describetag :tmp_dir

    setup [:setup_iris_model]

    test "returns predicted values case single x", %{
      tmp_dir: tmp_dir
    } do
      model = LgbmEx.load_model(tmp_dir, "iris")
      {x_test, _y} = SampleDataIris.test_set()
      features = List.first(x_test)

      [c1_prob, c2_prob, c3_prob] = LgbmEx.predict(model, features)
      assert c1_prob >= 0.5
      assert c2_prob >= 0.0
      assert c3_prob >= 0.0
    end

    test "returns predicted values case multi x", %{
      tmp_dir: tmp_dir
    } do
      model = LgbmEx.load_model(tmp_dir, "iris")
      {x_test, _y} = SampleDataIris.test_set()

      [[c1_prob, c2_prob, c3_prob] | _] = results = LgbmEx.predict(model, x_test)
      assert c1_prob >= 0.5
      assert c2_prob >= 0.0
      assert c3_prob >= 0.0
      assert Enum.count(results) == Enum.count(x_test)
    end
  end

  describe "save_as" do
    @describetag :tmp_dir

    test "returns model saved", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()
      model = LgbmEx.new_model(tmp_dir)
      model = LgbmEx.fit(model, x, y, parameters)

      saved_model = LgbmEx.save_as(model, "iris")
      assert File.exists?(saved_model.files.model)
    end
  end

  describe "load_model" do
    @describetag :tmp_dir

    setup [:setup_iris_model]

    test "returns model loaded", %{
      tmp_dir: tmp_dir
    } do
      loaded_model = LgbmEx.load_model(tmp_dir, "iris")
      assert %{metric: "multi_logloss"} = Map.new(loaded_model.parameters)
    end
  end

  describe "dump_zip" do
    @describetag :tmp_dir

    setup [:setup_iris_model]

    test "returns model loaded", %{
      tmp_dir: tmp_dir
    } do
      model = LgbmEx.load_model(tmp_dir, "iris")
      zip_path = LgbmEx.dump_zip(model)

      assert File.exists?(zip_path)
      assert String.ends_with?(zip_path, "/iris.zip")
    end
  end

  describe "from_zip" do
    @describetag :tmp_dir

    setup [:setup_iris_model]

    test "returns model loaded", %{
      tmp_dir: tmp_dir
    } do
      model = LgbmEx.load_model(tmp_dir, "iris")
      zip_path = LgbmEx.dump_zip(model)

      model = LgbmEx.from_zip(zip_path, tmp_dir, "iris_from_zip")

      assert model.name == "iris_from_zip"
      assert model.num_iterations == 10

      {x_test, _y} = SampleDataIris.test_set()
      features = List.first(x_test)

      [c1_prob | _] = LgbmEx.predict(model, features)
      assert c1_prob >= 0.5
    end
  end
end
