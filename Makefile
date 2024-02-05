# Environment variables from elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

TEMP ?= $(HOME)/.cache
LGBM_CACHE ?= $(TEMP)/LightGBM
LGBM_GIT_REPO ?= https://github.com/microsoft/LightGBM.git

# 4.3.0 Release Commit
LGBM_GIT_REV ?= 252828fd86627d7405021c3377534d6a8239dd69

# microsoft/LightGBM directories (in cache directory)
LGBM_NS = light-gbm-$(LGBM_GIT_REV)
LGBM_DIR = $(LGBM_CACHE)/$(LGBM_NS)
LGBM_LIB_DIR = $(LGBM_DIR)/build/light-gbm
LGBM_LIB_DIR_FLAG = $(LGBM_LIB_DIR)/light-gbm.ok

# LgbmEx files
PRIV_DIR = $(MIX_APP_PATH)/priv
LGBM_EX_DIR = $(realpath c_src/)
LGBM_EX_CACHE_SO = cache/liblgbmex.so
LGBM_EX_CACHE_LIB_DIR = cache/lib
LGBM_EX_SO = $(PRIV_DIR)/liblgbmex.so
LGBM_EX_LIB_DIR = $(PRIV_DIR)/lib

# Build Flags
C_SRCS = $(wildcard $(LGBM_EX_DIR)/*.cpp) $(wildcard $(LGBM_EX_DIR)/include/*.h)

CFLAGS = -I$(LGBM_EX_DIR)/include -I$(LGBM_DIR)/include -I$(LGBM_DIR) -I$(ERTS_INCLUDE_DIR) -std=c++17 -fpic -fpermissive --verbose -shared

LDFLAGS = -L$(LGBM_EX_CACHE_LIB_DIR) -l_lightgbm

# Linux only
LIBLGBM = lib_lightgbm.so
LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
LDFLAGS += -Wl,--allow-multiple-definition
POST_INSTALL = $(NOOP)

# Cache to priv
$(LGBM_EX_SO): $(LGBM_EX_CACHE_SO)
	@ mkdir -p $(PRIV_DIR)
	cp -a $(abspath $(LGBM_EX_CACHE_LIB_DIR)) $(LGBM_EX_LIB_DIR) ; \
	cp -a $(abspath $(LGBM_EX_CACHE_SO)) $(LGBM_EX_SO) ;

# Make LgbmEx library
$(LGBM_EX_CACHE_SO): $(LGBM_LIB_DIR_FLAG) $(C_SRCS)
	@mkdir -p cache
	cp -a $(LGBM_LIB_DIR) $(LGBM_EX_CACHE_LIB_DIR)
	cp $(LGBM_EX_CACHE_LIB_DIR)/lib/$(LIBLGBM) $(LGBM_EX_CACHE_LIB_DIR)
	$(CC) $(CFLAGS) $(wildcard $(LGBM_EX_DIR)/*.cpp) $(LDFLAGS) -o $(LGBM_EX_CACHE_SO)
	$(POST_INSTALL)

# Make microsoft/LightGBM
$(LGBM_LIB_DIR_FLAG):
		rm -rf $(LGBM_DIR) && \
		mkdir -p $(LGBM_DIR) && \
			cd $(LGBM_DIR) && \
			git init && \
			git remote add origin $(LGBM_GIT_REPO) && \
			git fetch --depth 1 --recurse-submodules origin $(LGBM_GIT_REV) && \
			git checkout FETCH_HEAD && \
			git submodule update --init --recursive && \
			cmake -DCMAKE_INSTALL_PREFIX=$(LGBM_LIB_DIR) -B build . $(CMAKE_FLAGS) && \
			make -C build  -j4 install
		touch $(LGBM_LIB_DIR_FLAG)

clean:
	rm -rf $(LGBM_EX_CACHE_SO)
	rm -rf $(LGBM_EX_CACHE_LIB_DIR)
	rm -rf $(LGBM_EX_SO)
	rm -rf $(LGBM_EX_LIB_DIR)
	rm -rf $(LGBM_DIR)
	rm -rf $(LGBM_LIB_DIR_FLAG)
