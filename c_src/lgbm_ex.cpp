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
    LightGBMModel() : booster_handle(nullptr), fastconfig_handle(nullptr) {}

    ~LightGBMModel() {
      if(booster_handle != nullptr) {
        LGBM_BoosterFree(booster_handle);
      }
      if(fastconfig && fastconfig_handle != nullptr) {
        LGBM_FastConfigFree(fastconfig_handle);
      }
    }

    // LGBM_BoosterCreateFromModelfile
    // Returns 0 on success, non-zero on error
    int booster_create_from_model_file(std::string filename, int* num_iteration) {
      int result = LGBM_BoosterCreateFromModelfile(
        filename.c_str(),
        num_iteration,
        &booster_handle
      );

      return result;
    }

    // LGBM_BoosterPredictForMatSingleRow
    // Returns 0 on success, non-zero on error
    int booster_predict_for_mat_single_row(json row, int num_features, int num_classes, std::vector<double>& out_result) {
      int64_t out_len;
      out_result.resize(num_classes, 0.0);
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
        int result = LGBM_BoosterPredictForMatSingleRowFastInit(
          booster_handle,
          C_API_PREDICT_NORMAL,
          0,
          0,
          C_API_DTYPE_FLOAT32,
          num_features,
          "",
          &fastconfig_handle
        );
        if (result != 0) return result;
        fastconfig = true;
      }

      int result = LGBM_BoosterPredictForMatSingleRowFast(
        fastconfig_handle,
        f_row.data(),
        &out_len,
        out_result.data()
      );

      return result;
    }

    // LGBM_BoosterPredictForMat
    // Returns 0 on success, non-zero on error
    int booster_predict_for_mat(json row, int nrow, int num_features, int num_classes, std::vector<double>& out_result) {
      int64_t out_len;
      out_result.resize(nrow * num_classes, 0.0);
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

      int result = LGBM_BoosterPredictForMat(
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
        &out_len,
        out_result.data()
      );

      return result;
    }

    // LGBM_BoosterGetNumClasses
    // Returns 0 on success, non-zero on error
    int booster_get_num_classes(int* num_classes) {
      int result = LGBM_BoosterGetNumClasses(
        booster_handle,
        num_classes
      );

      return result;
    }

    // LGBM_BoosterGetNumFeatures
    // Returns 0 on success, non-zero on error
    int booster_get_num_features(int* num_features) {
      int result = LGBM_BoosterGetNumFeature(
        booster_handle,
        num_features
      );

      return result;
    }

    // LGBM_BoosterGetCurrentIteration
    // Returns 0 on success, non-zero on error
    int booster_get_current_iteration(int* current_iteration) {
      int result = LGBM_BoosterGetCurrentIteration(
        booster_handle,
        current_iteration
      );

      return result;
    }

    // LGBM_BoosterGetEval
    // Returns 0 on success, non-zero on error
    int booster_get_eval(std::vector<double>& out_result) {
      int result;
      int num_result;
      int eval_count = 0;

      result = LGBM_BoosterGetEvalCounts(booster_handle, &eval_count);
      if (result != 0) return result;

      out_result.resize(eval_count, 0.0);

      result = LGBM_BoosterGetEval(
        booster_handle,
        0,
        &num_result,
        out_result.data()
      );

      return result;
    }

    // LGBM_BoosterGetLoadedParam
    // Returns 0 on success, non-zero on error
    int booster_get_loaded_param(std::string& out_param) {
      int64_t out_len;
      int64_t buf_len = 1024 * 1024;
      std::vector<char> out_str(buf_len);

      int result = LGBM_BoosterGetLoadedParam(
        booster_handle,
        buf_len,
        &out_len,
        out_str.data()
      );

      if (result == 0) {
        out_param = std::string(out_str.data());
      }

      return result;
    }

    // LGBM_BoosterFeatureImportanceGain
    // Returns 0 on success, non-zero on error
    int booster_feature_importance_gain(int iteration, int num_features, std::vector<double>& out_result) {
      out_result.resize(num_features, 0.0);

      int result = LGBM_BoosterFeatureImportance(
        booster_handle,
        iteration,
        C_API_FEATURE_IMPORTANCE_GAIN,
        out_result.data()
      );

      return result;
    }

    // LGBM_BoosterFeatureImportanceSplit
    // Returns 0 on success, non-zero on error
    int booster_feature_importance_split(int iteration, int num_features, std::vector<double>& out_result) {
      out_result.resize(num_features, 0.0);

      int result = LGBM_BoosterFeatureImportance(
        booster_handle,
        iteration,
        C_API_FEATURE_IMPORTANCE_SPLIT,
        out_result.data()
      );

      return result;
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

  // Use std::vector for automatic memory management (RAII pattern)
  std::vector<char> arg_json(len_f);
  enif_get_string(env, arg, arg_json.data(), len_f, ERL_NIF_LATIN1);

  return json::parse(arg_json.data());
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
  try {
    void* resource = enif_alloc_resource(ResourceType, sizeof(LightGBMModel));
    ERL_NIF_TERM handle = enif_make_resource(env, resource);
    LightGBMModel* model = new(resource) LightGBMModel;

    json j = decode_json(env, argv[0]);
    int num_iteration;
    int result = model->booster_create_from_model_file(j["file"], &num_iteration);

    if (result != 0) {
      // Error occurred - return {:error, error_message}
      const char* error_msg = LGBM_GetLastError();
      enif_release_resource(resource);
      return enif_make_tuple2(
        env,
        enif_make_atom(env, "error"),
        enif_make_string(env, error_msg, ERL_NIF_LATIN1)
      );
    }

    enif_release_resource(resource);

    // Success - return {:ok, handle}
    return enif_make_tuple2(
      env,
      enif_make_atom(env, "ok"),
      handle
    );
  } catch (const std::exception& e) {
    return enif_make_tuple2(
      env,
      enif_make_atom(env, "error"),
      enif_make_string(env, (std::string("Exception: ") + e.what()).c_str(), ERL_NIF_LATIN1)
    );
  } catch (...) {
    return enif_make_tuple2(
      env,
      enif_make_atom(env, "error"),
      enif_make_string(env, "Unknown exception occurred", ERL_NIF_LATIN1)
    );
  }
}

ERL_NIF_TERM booster_predict_for_mat_single_row(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    json j = decode_json(env, argv[1]);
    std::vector<double> result;
    int num_classes, num_features;

    int ret_code = model->booster_get_num_classes(&num_classes);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_get_num_features(&num_features);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_predict_for_mat_single_row(
      j["row"],
      num_features,
      num_classes,
      result
    );
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["num_features"] = num_features;
    ret_j["result"] = result;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_predict_for_mat(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    json j = decode_json(env, argv[1]);
    std::vector<double> result;
    int num_classes, num_features;

    int ret_code = model->booster_get_num_classes(&num_classes);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_get_num_features(&num_features);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_predict_for_mat(
      j["row"],
      j["nrow"],
      num_features,
      num_classes,
      result
    );
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["num_classes"] = num_classes;
    ret_j["num_features"] = num_features;
    ret_j["result"] = split_vector(result, j["nrow"], num_classes);

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_get_num_classes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    int num_classes;
    int ret_code = model->booster_get_num_classes(&num_classes);

    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["result"] = num_classes;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_get_num_features(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    int num_features;
    int ret_code = model->booster_get_num_features(&num_features);

    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["result"] = num_features;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_get_current_iteration(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    int current_iteration;
    int ret_code = model->booster_get_current_iteration(&current_iteration);

    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["result"] = current_iteration;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_get_eval(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);
    std::vector<double> result;

    int ret_code = model->booster_get_eval(result);

    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["result"] = result;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_get_loaded_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);

    std::string result;
    int ret_code = model->booster_get_loaded_param(result);

    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["result"] = result;

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
}

ERL_NIF_TERM booster_feature_importance(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  try {
    LightGBMModel* model = load_model(env, argv[0]);
    int num_features, iteration;

    int ret_code = model->booster_get_num_features(&num_features);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_get_current_iteration(&iteration);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    std::vector<double> result_split, result_gain;

    ret_code = model->booster_feature_importance_split(iteration, num_features, result_split);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    ret_code = model->booster_feature_importance_gain(iteration, num_features, result_gain);
    if (ret_code != 0) {
      json err_j;
      err_j["error"] = LGBM_GetLastError();
      return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
    }

    json ret_j;
    ret_j["iteration"] = iteration;
    ret_j["num_features"] = num_features;
    ret_j["result"] = {result_split, result_gain};

    return enif_make_string(env, ret_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (const std::exception& e) {
    json err_j;
    err_j["error"] = std::string("Exception: ") + e.what();
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  } catch (...) {
    json err_j;
    err_j["error"] = "Unknown exception occurred";
    return enif_make_string(env, err_j.dump().c_str(), ERL_NIF_LATIN1);
  }
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
  {"booster_feature_importance", 1, booster_feature_importance}
};

ERL_NIF_INIT(Elixir.LgbmEx.NIF, nif_funcs, nif_load, nullptr, nullptr, nullptr);
