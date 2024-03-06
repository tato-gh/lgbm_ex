#include <erl_nif.h>
#include <LightGBM/c_api.h>
#include <new>
#include <vector>
#include <string>
#include <map>

#include "json.hpp"
using json = nlohmann::json;

class LightGBMModel {
  public:
    LightGBMModel() {}

    ~LightGBMModel() {
      LGBM_BoosterFree(booster_handle);
      if(fastconfig) {
        LGBM_FastConfigFree(fastconfig_handle);
      }
    }

    // LGBM_BoosterCreateFromModelfile
    int booster_create_from_model_file(std::string filename) {
      int result;
      int num_iteration;

      result = LGBM_BoosterCreateFromModelfile(
        filename.c_str(),
        &num_iteration,
        &booster_handle
      );

      return num_iteration;
    }

    // LGBM_BoosterPredictForMatSingleRow
    std::vector<double> booster_predict_for_mat_single_row(json row, int num_features, int num_classes) {
      int64_t *out_len = new int64_t();
      std::vector<double> out_result(num_classes, 0.0);
      std::vector<float> f_row;
      for(auto v: row) {
        if(v.is_null()) {
          f_row.push_back(NAN);
        } else if(v.is_string()) {
          std::string str = v;
          f_row.push_back(std::stof(str));
        } else {
          f_row.push_back(v);
        }
      }

      if(fastconfig == false) {
        LGBM_BoosterPredictForMatSingleRowFastInit(
          booster_handle,
          C_API_PREDICT_NORMAL,
          0,
          0,
          C_API_DTYPE_FLOAT32,
          num_features,
          "",
          &fastconfig_handle
        );
        fastconfig = true;
      }

      LGBM_BoosterPredictForMatSingleRowFast(
        fastconfig_handle,
        f_row.data(),
        out_len,
        out_result.data()
      );

      return out_result;
    }

    // LGBM_BoosterPredictForMat
    std::vector<double> booster_predict_for_mat(json row, int nrow, int num_features, int num_classes) {
      int64_t *out_len = new int64_t();
      std::vector<double> out_result(nrow * num_classes, 0.0);
      std::vector<float> f_row;
      for(auto v: row) {
        if(v.is_null()) {
          f_row.push_back(NAN);
        } else if(v.is_string()) {
          std::string str = v;
          f_row.push_back(std::stof(str));
        } else {
          f_row.push_back(v);
        }
      }

      LGBM_BoosterPredictForMat(
        booster_handle,
        f_row.data(),
        C_API_DTYPE_FLOAT32,
        nrow,
        num_features,
        1,
        C_API_PREDICT_NORMAL,
        0,
        0,
        "",
        out_len,
        out_result.data()
      );

      return out_result;
    }

    // LGBM_BoosterGetNumClasses
    int booster_get_num_classes() {
      int result;
      int num_classes;

      result = LGBM_BoosterGetNumClasses(
        booster_handle,
        &num_classes
      );

      return num_classes;
    }

    // LGBM_BoosterGetNumFeatures
    int booster_get_num_features() {
      int result;
      int num_features;

      result = LGBM_BoosterGetNumFeature(
        booster_handle,
        &num_features
      );

      return num_features;
    }

    // LGBM_BoosterGetCurrentIteration
    int booster_get_current_iteration() {
      int result;
      int current_iteration;

      result = LGBM_BoosterGetCurrentIteration(
        booster_handle,
        &current_iteration
      );

      return current_iteration;
    }

    // LGBM_BoosterGetEval
    std::vector<double> booster_get_eval() {
      int result;
      int num_result;
      int eval_count = 0;

      LGBM_BoosterGetEvalCounts(booster_handle, &eval_count);

      std::vector<double> out_result(eval_count, 0.0);

      result = LGBM_BoosterGetEval(
        booster_handle,
        0,
        &num_result,
        out_result.data()
      );

      return out_result;
    }

    // LGBM_BoosterGetLoadedParam
    std::string booster_get_loaded_param() {
      int64_t *out_len = new int64_t();
      int64_t buf_len = 1024 * 1024;
      char* out_str = (char*)malloc(buf_len * sizeof(char));
      int result;

      result = LGBM_BoosterGetLoadedParam(
        booster_handle,
        buf_len,
        out_len,
        out_str
      );

      return std::string(out_str);
    }

    // LGBM_BoosterFeatureImportanceGain
    std::vector<double> booster_feature_importance_gain(int iteration, int num_features) {
      std::vector<double> out_result(num_features, 0.0);
      int result;

      result = LGBM_BoosterFeatureImportance(
        booster_handle,
        iteration,
        C_API_FEATURE_IMPORTANCE_GAIN,
        out_result.data()
      );

      return out_result;
    }

    // LGBM_BoosterFeatureImportanceSplit
    std::vector<double> booster_feature_importance_split(int iteration, int num_features) {
      std::vector<double> out_result(num_features, 0.0);
      int result;

      result = LGBM_BoosterFeatureImportance(
        booster_handle,
        iteration,
        C_API_FEATURE_IMPORTANCE_SPLIT,
        out_result.data()
      );

      return out_result;
    }

  private:
    BoosterHandle booster_handle;
    FastConfigHandle fastconfig_handle;
    bool fastconfig = false;
};

ErlNifResourceType* ResourceType;

LightGBMModel* load_model(ErlNifEnv* env, const ERL_NIF_TERM arg) {
  void* resource;
  enif_get_resource(env, arg, ResourceType, &resource);

  return static_cast<LightGBMModel*>(resource);
}

void destruct(ErlNifEnv* caller_env, void* obj) {
  LightGBMModel* model = static_cast<LightGBMModel*>(obj);
  model->~LightGBMModel();
}

int nif_load(ErlNifEnv* caller_env, void** priv_data, ERL_NIF_TERM load_info) {
  ResourceType = enif_open_resource_type(caller_env, "Elixir.LGBMExCapi", "Interface", destruct, ERL_NIF_RT_CREATE, nullptr);

  return 0;
}

json decode_json(ErlNifEnv* env, const ERL_NIF_TERM arg) {
  unsigned len_f;
  enif_get_list_length(env, arg, &len_f);
  len_f++;
  char *arg_json = reinterpret_cast<char*>(enif_alloc(len_f));
  enif_get_string(env, arg, arg_json, len_f, ERL_NIF_LATIN1);

  return json::parse(arg_json);
}

std::vector<std::vector<double>> split_vector(std::vector<double> data, int nrow, int ncol) {
  std::vector<std::vector<double>> arr(nrow, std::vector<double>(ncol));

  for(int i = 0; i < nrow; i++) {
    for(int j = 0; j < ncol; j++) {
      arr[i][j] = data[i * ncol + j];
    }
  }

  return arr;
}

// APIs
// ==========================

ERL_NIF_TERM booster_create_from_model_file(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  void* resource = enif_alloc_resource(ResourceType, sizeof(LightGBMModel));
  ERL_NIF_TERM handle = enif_make_resource(env, resource);
  LightGBMModel* model = new(resource) LightGBMModel;

  json j = decode_json(env, argv[0]);
  int result;

  result = model->booster_create_from_model_file(j["file"]);

  enif_release_resource(resource);

  // Segmentation fault is occured when returns tuple3 {:ok, result, handle}
  return enif_make_tuple2(
    env,
    enif_make_atom(env, "ok"),
    handle
  );
}

ERL_NIF_TERM booster_predict_for_mat_single_row(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  json j = decode_json(env, argv[1]);
  std::vector<double> result;

  int num_classes = model->booster_get_num_classes();
  int num_features = model->booster_get_num_features();
  result = model->booster_predict_for_mat_single_row(
    j["row"],
    num_features,
    num_classes
  );

  json ret_j;
  ret_j["num_features"] = num_features;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_predict_for_mat(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  json j = decode_json(env, argv[1]);
  std::vector<double> result;

  int num_classes = model->booster_get_num_classes();
  int num_features = model->booster_get_num_features();
  result = model->booster_predict_for_mat(
    j["row"],
    j["nrow"],
    num_features,
    num_classes
  );

  json ret_j;
  ret_j["num_classes"] = num_classes;
  ret_j["num_features"] = num_features;
  ret_j["result"] = split_vector(result, j["nrow"], num_classes);

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_get_num_classes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  int result = model->booster_get_num_classes();

  json ret_j;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_get_num_features(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  int result = model->booster_get_num_features();

  json ret_j;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_get_current_iteration(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  int result = model->booster_get_current_iteration();

  json ret_j;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_get_eval(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);
  std::vector<double> result;

  result = model->booster_get_eval();

  json ret_j;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_get_loaded_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);

  std::string result = model->booster_get_loaded_param();

  json ret_j;
  ret_j["result"] = result;

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_feature_importance(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  LightGBMModel* model = load_model(env, argv[0]);
  int num_features = model->booster_get_num_features();
  int iteration = model->booster_get_current_iteration();

  std::vector<double> result_split;
  std::vector<double> result_gain;
  result_split = model->booster_feature_importance_split(iteration, num_features);
  result_gain = model->booster_feature_importance_gain(iteration, num_features);

  json ret_j;
  ret_j["iteration"] = iteration;
  ret_j["num_features"] = num_features;
  ret_j["result"] = {result_split, result_gain};

  return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM booster_free(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  enif_free_env(env);

  ERL_NIF_TERM ret = -1;
  return ret;
}

static ErlNifFunc nif_funcs[] = {
  {"booster_create_from_model_file", 1, booster_create_from_model_file},
  {"booster_predict_for_mat_single_row", 2, booster_predict_for_mat_single_row},
  {"booster_predict_for_mat", 2, booster_predict_for_mat},
  {"booster_get_num_classes", 1, booster_get_num_classes},
  {"booster_get_num_features", 1, booster_get_num_features},
  {"booster_get_current_iteration", 1, booster_get_current_iteration},
  {"booster_get_eval", 1, booster_get_eval},
  {"booster_get_loaded_param", 1, booster_get_loaded_param},
  {"booster_feature_importance", 1, booster_feature_importance},
  {"booster_free", 1, booster_free}
};

ERL_NIF_INIT(Elixir.LgbmEx.NIF, nif_funcs, nif_load, nullptr, nullptr, nullptr);
