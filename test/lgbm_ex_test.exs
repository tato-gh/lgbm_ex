defmodule LgbmExTest do
  use ExUnit.Case
  doctest LgbmEx

  alias LgbmEx.SampleDataIris

  def setup_iris_model(%{tmp_dir: tmp_dir}) do
    {x, y} = SampleDataIris.train_set()
    parameters = SampleDataIris.parameters()
    model = LgbmEx.new_model(tmp_dir)
    {model, _, _} = LgbmEx.fit(model, x, y, parameters)
    LgbmEx.save_as(model, "iris")
    :ok
  end

  describe "fit" do
    @describetag :tmp_dir

    test "returns model, num of iterations", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      {model, num_iterations, learning_steps} = LgbmEx.fit(model, x, y, parameters)

      assert model
      assert num_iterations == 10
      # cannot get values because of unuse early_stopping
      assert learning_steps == []
    end

    test "early stopping and returns steps", %{
      tmp_dir: tmp_dir
    } do
      {x_train, y_train} = SampleDataIris.train_set()
      {x_val, y_val} = SampleDataIris.test_set()
      parameters = SampleDataIris.parameters_with_early_stopping()

      model = LgbmEx.new_model(tmp_dir)
      {model, num_iterations, learning_steps} = LgbmEx.fit(model, {x_train, x_val}, {y_train, y_val}, parameters)

      assert model
      assert num_iterations > 10
      assert Enum.count(learning_steps) > 10
      assert hd(learning_steps) == {0, 0.939663}
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
      {model, _num_iterations, _learning_steps} = LgbmEx.fit(model, x, y, parameters)

      {_model, num_iterations, _learning_steps} = LgbmEx.refit(model, num_iterations: 2)
      assert num_iterations == 2
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
      {model, _, _} = LgbmEx.fit(model, x, y, parameters)

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

  describe "predict" do
    @describetag :tmp_dir

    setup [:setup_iris_model]

    test "returns predicted values, case single x", %{
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
  end
end
