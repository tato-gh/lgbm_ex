defmodule LgbmExTest do
  use ExUnit.Case
  doctest LgbmEx

  alias LgbmEx.SampleDataIris

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

  describe "save_as" do
    @describetag :tmp_dir

    test "returns model saved", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()
      model = LgbmEx.new_model(tmp_dir)
      {model, _, _} = LgbmEx.fit(model, x, y, parameters)

      saved_model = LgbmEx.save_as(model, "hoge")
      assert File.exists?(saved_model.files.model)
    end
  end
end

