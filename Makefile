CXX ?= $(shell command -v g++ || shell command -v clang++)
AR ?= ar
PY := $(shell command -v python3 || shell command -v python)

ifeq ($(CXX),)
  $(error "C++ compiler '$(CXX)' not found. Please install g++ or clang.")
endif
ifeq ($(AR),)
  $(error "Archiver '$(AR)' not found. Please install build-essential or binutils.")
endif
ifeq ($(PY),)
  $(error "Python not found. Please install Python or create a virtualenv first (e.g., 'python3 -m venv venv && source venv/bin/activate').")
endif

SOLVER ?= bitwuzla
INSTALL_PREFIX ?= /usr/local

CXXFLAGS = -std=c++20 -Iinclude -Wall -Wextra -O2
LDFLAGS =
ARFLAGS = rcs

# Coverage support
ifeq ($(COVERAGE), 1)
  CXXFLAGS += --coverage
  LDFLAGS += --coverage
endif

# Solver support
ifeq ($(SOLVER), bitwuzla)
  ifneq ($(shell pkg-config --exists bitwuzla && echo found),found)
    $(error "Bitwuzla not found. Please install it and ensure it is discoverable via pkg-config.")
  endif
  CXXFLAGS += -DUSE_BITWUZLA $(shell pkg-config --cflags bitwuzla)
  LDFLAGS += $(shell pkg-config --libs bitwuzla)
  SOLVER_SRCS += src/solver/bitwuzla_impl.cpp
  SOLVER_IMPL_OBJ = src/solver/bitwuzla_impl.o
else ifeq ($(SOLVER), alivesmt)
  ifneq ($(shell pkg-config --exists z3 && echo found),found)
    $(error "Z3 not found. Please install it and ensure it is discoverable via pkg-config.")
  endif
  CXXFLAGS += -DUSE_ALIVESMT -Ialivesmt/include $(shell pkg-config --cflags z3)
  LDFLAGS += $(shell pkg-config --libs z3)
  ALIVESMT_SRCS = alivesmt/lib/ctx.cpp alivesmt/lib/expr.cpp alivesmt/lib/exprs.cpp alivesmt/lib/smt.cpp alivesmt/lib/solver.cpp alivesmt/lib/util.cpp
  SOLVER_SRCS += src/solver/alive_impl.cpp $(ALIVESMT_SRCS)
  SOLVER_IMPL_OBJ = src/solver/alive_impl.o $(ALIVESMT_SRCS:.cpp=.o)
else
  $(error "Unknown SOLVER: $(SOLVER). Supported: bitwuzla, alivesmt")
endif

COMMON_SRCS = src/frontend/lexer.cpp src/frontend/parser.cpp src/frontend/ast_dumper.cpp \
              src/frontend/sir_printer.cpp src/frontend/pipeline.cpp src/ast/clone.cpp \
              src/analysis/cfgbuilder.cpp src/analysis/definite_init.cpp \
              src/analysis/dominators.cpp src/analysis/reducibility.cpp \
              src/analysis/loop_info.cpp src/analysis/structurizer.cpp \
              src/analysis/structured_lowering.cpp \
              src/frontend/typechecker.cpp src/frontend/semchecker.cpp \
              src/analysis/pass_manager.cpp src/analysis/reachability.cpp \
              src/analysis/unused_name.cpp src/analysis/type_utils.cpp \
              src/analysis/state_set.cpp \
              src/frontend/diagnostics.cpp \
              src/solver/solved_header.cpp

TEST_SRCS =
BACKEND_SRCS = src/backend/emit.cpp \
               src/backend/c_backend.cpp src/backend/wasm_backend.cpp \
               src/backend/c_expr.cpp src/backend/c_vec.cpp \
               src/backend/c_lvalue.cpp src/backend/c_types.cpp \
               src/backend/c_structured.cpp \
               src/backend/wasm_expr.cpp src/backend/wasm_vec.cpp \
               src/backend/wasm_structured.cpp \
               src/backend/wasm_lvalue.cpp src/backend/wasm_types.cpp \
               src/backend/wasm_vec_lowering_array.cpp \
               src/backend/wasm_vec_lowering_scalars.cpp \
               src/backend/wasm_vec_lowering_vecext.cpp \
               src/backend/py_backend.cpp src/backend/py_expr.cpp \
               src/backend/py_types.cpp src/backend/py_lvalue.cpp \
               src/backend/py_vec_lowering_array.cpp \
               src/backend/py_vec_lowering_scalars.cpp \
               src/backend/py_vec_lowering_struct.cpp \
               src/backend/vec_shapes.cpp \
               src/backend/c_intrinsics.cpp src/backend/wasm_intrinsics.cpp \
               src/backend/py_intrinsics.cpp \
               src/backend/c_vec_lowering_vecext.cpp \
               src/backend/c_vec_lowering_array.cpp \
               src/backend/c_vec_lowering_scalars.cpp \
               src/backend/c_vec_lowering_struct.cpp
INTERP_IMPL_SRCS = src/interp/interpreter.cpp src/interp/intrinsics.cpp \
                   src/interp/type_layout.cpp src/interp/types.cpp \
                   src/interp/memory.cpp src/interp/values.cpp \
                   src/interp/expr.cpp src/interp/lvalue.cpp
INTERP_SRCS = src/symiri.cpp $(INTERP_IMPL_SRCS)
COMPILER_SRCS = src/symirc.cpp $(BACKEND_SRCS)
# Core symbolic-executor sources (distinct from the SMT backend in
# SOLVER_SRCS / SOLVER_IMPL_OBJ). Shared by symirsolve, rysmith, rylink, and
# libsymir, so every consumer picks up new TUs from one place.
SOLVER_CORE_SRCS = src/solver/solver.cpp src/solver/intrinsics.cpp \
                   src/solver/values.cpp src/solver/expr.cpp \
                   src/solver/vec.cpp src/solver/lvalue.cpp \
                   src/solver/calls.cpp src/solver/types.cpp
SOLVER_MAIN_SRCS = src/symirsolve.cpp $(SOLVER_CORE_SRCS)
SOLVER_ALL_SRCS = $(SOLVER_MAIN_SRCS) $(SOLVER_SRCS)
REIFY_SRCS = src/reify/cfg_gen.cpp src/reify/path_sampler.cpp \
             src/reify/type_gen.cpp src/reify/var_catalogue.cpp \
             src/reify/expr_gen.cpp src/reify/func_gen.cpp \
             src/reify/func_desc.cpp \
             src/reify/checksum.cpp \
             src/reify/common.cpp \
             src/reify/func_pool.cpp src/reify/cg_gen.cpp \
             src/reify/call_realize.cpp src/reify/dataflow_policy.cpp src/reify/name_alloc.cpp src/reify/state_profile.cpp \
             src/reify/transform.cpp src/reify/twin_transform.cpp \
             src/reify/twin_mini.cpp \
             src/reify/twin_probe.cpp src/reify/twin_trace.cpp \
             src/reify/twin_box.cpp src/reify/antiopt.cpp \
             src/reify/antiopt/common.cpp src/reify/antiopt/rules_peephole.cpp \
             src/reify/antiopt/rules_mba.cpp src/reify/antiopt/rules_structure.cpp \
             src/reify/antiopt/rules_licensed.cpp \
             src/reify/antiopt/rules_control.cpp \
             src/reify/antiopt/selftest.cpp src/reify/antiopt_transform.cpp
RYSMITH_SRCS = src/rysmith.cpp $(SOLVER_CORE_SRCS) $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)
# [v0.2.2] rylink links the C / WASM backends in-process so the bundle's
# FunDecl::sourceStem survives all the way to emitSplit. Driving symirc
# as a subprocess instead would parse the bundled .sir back from text
# and reset every sourceStem to "", collapsing --split-by-source to a
# single program.c.
RYLINK_SRCS = src/rylink.cpp $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)
# rytwin builds twin bodies from the profiled trace and decides guards by
# interpretation, so it links neither the symbolic executor nor an SMT
# backend — unlike rysmith and rylink.
RYTWIN_SRCS = src/rytwin.cpp $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)

COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
INTERP_OBJS = $(INTERP_SRCS:.cpp=.o)
COMPILER_OBJS = $(COMPILER_SRCS:.cpp=.o)
SOLVER_OBJS = $(SOLVER_MAIN_SRCS:.cpp=.o) $(SOLVER_IMPL_OBJ)
RYSMITH_OBJS = $(RYSMITH_SRCS:.cpp=.o) $(SOLVER_IMPL_OBJ)
RYLINK_OBJS = $(RYLINK_SRCS:.cpp=.o)
RYTWIN_OBJS = $(RYTWIN_SRCS:.cpp=.o)

TARGET_INTERP = symiri
TARGET_COMPILER = symirc
TARGET_SOLVER = symirsolve
TARGET_RYSMITH = rysmith
TARGET_RYLINK = rylink
TARGET_RYTWIN = rytwin
TARGET_RYPUZMK = rypuzmk
TARGET_RYPUZCHK = rypuzchk
TARGET_RYPUZMK_SIR = rypuzmk-sir
TARGET_RYPUZCHK_SIR = rypuzchk-sir
TARGET_RYPUZMK_TGT = rypuzmk-tgt
TARGET_RYPUZCHK_TGT = rypuzchk-tgt
TARGET_CODOKU = codoku

BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
LIB_DIR = $(BUILD_DIR)/lib
INC_DIR = $(BUILD_DIR)/include

LIB_NAME = libsymir.a
LIBRARY_OBJS = $(COMMON_OBJS) \
               src/interp/interpreter.o \
               src/interp/intrinsics.o \
               src/interp/type_layout.o \
               src/interp/types.o \
               src/interp/memory.o \
               src/interp/values.o \
               src/interp/expr.o \
               src/interp/lvalue.o \
               src/backend/c_backend.o \
               src/backend/c_expr.o \
               src/backend/c_vec.o \
               src/backend/c_lvalue.o \
               src/backend/c_types.o \
               src/backend/c_intrinsics.o \
               src/backend/wasm_intrinsics.o \
               src/backend/c_vec_lowering_vecext.o \
               src/backend/c_vec_lowering_array.o \
               src/backend/c_vec_lowering_scalars.o \
               src/backend/c_vec_lowering_struct.o \
               src/backend/wasm_backend.o \
               src/backend/wasm_expr.o \
               src/backend/wasm_vec.o \
               src/backend/wasm_structured.o \
               src/backend/wasm_lvalue.o \
               src/backend/wasm_types.o \
               src/backend/wasm_vec_lowering_array.o \
               src/backend/wasm_vec_lowering_scalars.o \
               src/backend/wasm_vec_lowering_vecext.o \
               src/backend/py_backend.o \
               src/backend/py_expr.o \
               src/backend/py_types.o \
               src/backend/py_lvalue.o \
               src/backend/py_vec_lowering_array.o \
               src/backend/py_vec_lowering_scalars.o \
               src/backend/py_vec_lowering_struct.o \
               src/backend/vec_shapes.o \
               src/backend/py_intrinsics.o \
               $(SOLVER_CORE_SRCS:.cpp=.o) \
               $(SOLVER_IMPL_OBJ)

.PHONY: all clean test test-unit test-frontend test-analysis test-interp test-backends test-cross-validation test-solver test-reify test-puzzle-sir test-puzzle-c cross-validation build install

all: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN) $(TARGET_RYPUZMK) $(TARGET_RYPUZCHK) $(TARGET_RYPUZMK_TGT) $(TARGET_RYPUZCHK_TGT) $(TARGET_RYPUZMK_SIR) $(TARGET_RYPUZCHK_SIR) $(TARGET_CODOKU)

$(TARGET_INTERP): $(COMMON_OBJS) $(INTERP_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_COMPILER): $(COMMON_OBJS) $(COMPILER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_SOLVER): $(COMMON_OBJS) $(SOLVER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYSMITH): $(COMMON_OBJS) $(RYSMITH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYPUZMK_SIR): puzzle/sir/rypuzmk.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYPUZCHK_SIR): puzzle/sir/rypuzchk.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYPUZMK_TGT): puzzle/target/rypuzmk.py
	ln -sf puzzle/target/rypuzmk.py $(TARGET_RYPUZMK_TGT)
	chmod +x $(TARGET_RYPUZMK_TGT)

$(TARGET_RYPUZCHK_TGT): puzzle/target/rypuzchk.py
	ln -sf puzzle/target/rypuzchk.py $(TARGET_RYPUZCHK_TGT)
	chmod +x $(TARGET_RYPUZCHK_TGT)

$(TARGET_RYPUZMK): puzzle/bin/rypuzmk
	ln -sf puzzle/bin/rypuzmk $(TARGET_RYPUZMK)
	chmod +x $(TARGET_RYPUZMK)

$(TARGET_RYPUZCHK): puzzle/bin/rypuzchk
	ln -sf puzzle/bin/rypuzchk $(TARGET_RYPUZCHK)
	chmod +x $(TARGET_RYPUZCHK)

$(TARGET_CODOKU): codokus/codoku.py
	ln -sf codokus/codoku.py $(TARGET_CODOKU)
	chmod +x $(TARGET_CODOKU)

# [v0.2.2] rylink also depends on the solver objects: it doesn't call
# the SMT solver itself, but the reify libs (and SIRPrinter) include
# solver headers and reference its types, so the linker needs the
# symbols. We don't link the Bitwuzla impl since rylink never solves.
$(TARGET_RYLINK): $(COMMON_OBJS) $(RYLINK_OBJS) $(SOLVER_CORE_SRCS:.cpp=.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYTWIN): $(COMMON_OBJS) $(RYTWIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build: all $(LIB_DIR)/$(LIB_NAME)
	mkdir -p $(BIN_DIR) $(INC_DIR)
	cp -f $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYPUZMK_SIR) $(TARGET_RYPUZCHK_SIR) $(TARGET_RYPUZMK) $(TARGET_RYPUZCHK) $(BIN_DIR)/
	cp -f puzzle/target/rypuzmk.py $(BIN_DIR)/$(TARGET_RYPUZMK_TGT)
	chmod +x $(BIN_DIR)/$(TARGET_RYPUZMK_TGT)
	cp -f puzzle/target/rypuzchk.py $(BIN_DIR)/$(TARGET_RYPUZCHK_TGT)
	chmod +x $(BIN_DIR)/$(TARGET_RYPUZCHK_TGT)
	cp -f puzzle/target/puzzle_common.py $(BIN_DIR)/puzzle_common.py
	cp -f codokus/codoku.py $(BIN_DIR)/$(TARGET_CODOKU)
	chmod +x $(BIN_DIR)/$(TARGET_CODOKU)
	cp -f codokus/codoku_creator.py $(BIN_DIR)/codoku_creator.py
	cp -f codokus/codoku_checker.py $(BIN_DIR)/codoku_checker.py
	cp -f codokus/codoku_common.py $(BIN_DIR)/codoku_common.py
	cp -f codokus/codoku_complexity.py $(BIN_DIR)/codoku_complexity.py
	cp -r include/* $(INC_DIR)/

install: build
	mkdir -p $(INSTALL_PREFIX)/bin
	mkdir -p $(INSTALL_PREFIX)/lib
	mkdir -p $(INSTALL_PREFIX)/include/refractir
	cp -fp $(BIN_DIR)/* $(INSTALL_PREFIX)/bin/
	cp -fp $(LIB_DIR)/* $(INSTALL_PREFIX)/lib/
	cp -rfp $(INC_DIR)/* $(INSTALL_PREFIX)/include/refractir/

$(LIB_DIR)/$(LIB_NAME): $(LIBRARY_OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -f $(COMMON_OBJS) $(TEST_OBJS) $(INTERP_OBJS) $(COMPILER_OBJS) $(SOLVER_OBJS) $(RYSMITH_OBJS) $(RYLINK_OBJS) $(RYTWIN_OBJS) puzzle/sir/rypuzmk.o puzzle/sir/rypuzchk.o $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN) $(TARGET_RYPUZMK) $(TARGET_RYPUZCHK) $(TARGET_RYPUZMK_TGT) $(TARGET_RYPUZCHK_TGT) $(TARGET_RYPUZMK_SIR) $(TARGET_RYPUZCHK_SIR) $(TARGET_CODOKU)
	rm -rf $(BUILD_DIR)
	find . -name "*.gcno" -delete
	find . -name "*.gcda" -delete
	find . -name "*.gcov" -delete

# [v0.2.2] Unit tests — end-to-end Python scripts that exercise the
# CLI surface of individual binaries (positional args, descriptors,
# split-by-source output, etc.). They don't go through the `.sir`
# test runner in test/lib because they need to assert on the binary's
# stdout / sidecar files / output directory layout.
test-unit: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN) $(TARGET_RYPUZMK) $(TARGET_RYPUZCHK) $(TARGET_RYPUZMK_TGT) $(TARGET_RYPUZCHK_TGT) $(TARGET_CODOKU)
	$(PY) -m test.unit.run_param_features_tests ./$(TARGET_INTERP) ./$(TARGET_COMPILER) ./$(TARGET_SOLVER)
	$(PY) -m test.unit.run_structured_c_tests ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_structured_wasm_tests ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_rysmith_tests ./$(TARGET_RYSMITH) ./$(TARGET_INTERP) ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_rylink_tests ./$(TARGET_RYLINK) ./$(TARGET_RYSMITH) ./$(TARGET_INTERP)
	$(PY) -m test.unit.run_rytwin_tests ./$(TARGET_RYTWIN) ./$(TARGET_RYSMITH) ./$(TARGET_INTERP)
	$(PY) -m test.unit.run_puzzle_sir_tests ./$(TARGET_RYPUZMK_SIR) ./$(TARGET_RYPUZCHK_SIR) ./$(TARGET_RYSMITH) ./$(TARGET_INTERP)
	$(PY) -m test.unit.run_puzzle_c_tests ./$(TARGET_RYPUZMK_TGT) ./$(TARGET_RYPUZCHK_TGT) ./$(TARGET_RYSMITH)
	$(PY) -m test.unit.run_puzzle_python_tests ./$(TARGET_RYPUZMK_TGT) ./$(TARGET_RYPUZCHK_TGT) ./$(TARGET_RYSMITH)
	$(PY) -m test.unit.run_codoku_tests ./$(TARGET_CODOKU) ./$(TARGET_RYSMITH)

# Integration tests, grouped by the component under test. Each component
# target is callable on its own (e.g. `make test-frontend`) so a developer
# touching one area can run just the relevant suite. `make test` invokes
# them in dependency order via `$(MAKE)` so a failure in an earlier group
# halts the run with the right exit code and the per-group banners stay
# legible in CI logs.

# Frontend: lexer, parser, CFG builder, type checker, sem checker, and
# reducibility analyses (dominator trees, loop forests, control trees).
test-frontend: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_interp_tests test/lexer ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/parser ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/cfgbuilder ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/typechecker ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/semchecker ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_reducibility_tests test/reducibility ./$(TARGET_COMPILER)

# Interpreter: end-to-end .sir execution.
test-interp: $(TARGET_INTERP)
	$(PY) -m test.lib.run_interp_tests test/interp ./$(TARGET_INTERP)

# Backends: C / WASM emission + the C preamble. Cross-validation lives in
# its own target below so it can be run on its own.
test-backends: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_c_preamble_test ./$(TARGET_COMPILER)
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target c
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target c --structured
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target wasm
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target wasm --structured
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target python

# Cross-validation: run every test/xval/*.sir through symiri *and* through
# the C backend + compiled-C and diff their behaviour. This is a separate
# group because it's the only suite that exercises the interpreter and the
# C backend together end-to-end, and a developer touching either side may
# want to run it on its own (`make cross-validation`).
test-cross-validation cross-validation: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_xval_tests test/xval ./$(TARGET_INTERP) ./$(TARGET_COMPILER)
	$(PY) -m test.lib.run_xval_tests test/xval ./$(TARGET_INTERP) ./$(TARGET_COMPILER) "--symirc-extra=--structured-lowering --no-ub-guards"

# Solver: symirsolve + curated examples that depend on solving.
test-solver: $(TARGET_SOLVER) $(TARGET_INTERP)
	$(PY) -m test.lib.run_solver_tests test/solver ./$(TARGET_SOLVER) ./$(TARGET_INTERP)
	$(PY) -m test.lib.run_example_tests examples ./$(TARGET_SOLVER) ./$(TARGET_INTERP)

# Reify: rysmith + rylink differential generation pipeline.
# Forward `make -jN` to the test runner's own `-j` flag so the
# per-program cross-validation phase (steps 2 and 4 of every batch)
# inherits the user's requested parallelism.  Generation and batching
# stay sequential inside the script regardless.
#
# MAKEFLAGS is empty at parse-time (GNU make only populates it when a
# recipe shell is spawned) so the extraction must run inside the
# recipe.  Inside the recipe: `make -j8` produces something like
# `-j8 --jobserver-auth=3,4`; bare `make -j` is `-j …`; plain `make`
# is empty.  Grep for `-j<digits>`, default to 1 if absent — bare
# `make -j` is mapped to 1 too, since "unlimited" without a concrete
# value is a footgun for a script that fans out subprocess workers.
test-reify: $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_INTERP) $(TARGET_COMPILER)
	@JOBS=$$(echo "$(MAKEFLAGS)" | grep -oE -- '-j[0-9]+' | head -1 | sed 's/-j//'); \
	JOBS=$${JOBS:-1}; \
	$(PY) -m test.lib.run_reify_diff_tests --rysmith ./$(TARGET_RYSMITH) --symiri ./$(TARGET_INTERP) --symirc ./$(TARGET_COMPILER) --rylink ./$(TARGET_RYLINK) -n 100 --seed 1234 -j $$JOBS

# Puzzle tooling: rypuzmk (maker) + rypuzchk (checker). Two suites:
test-puzzle-sir: $(TARGET_RYPUZMK) $(TARGET_RYPUZCHK) $(TARGET_RYSMITH) $(TARGET_INTERP)
	$(PY) -m test.lib.run_puzzle_sir_test test/puzzle-sir ./$(TARGET_RYPUZCHK_SIR) ./$(TARGET_INTERP)

test-puzzle-c: $(TARGET_RYPUZMK_TGT) $(TARGET_RYPUZCHK_TGT) $(TARGET_RYSMITH)
	$(PY) -m test.lib.run_puzzle_c_test test/puzzle-c ./$(TARGET_RYPUZCHK_TGT)

# Aggregator. Recursive $(MAKE) calls keep per-component logs separated and
# let CI selectively re-run a single group on retry.
test: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK)
	$(MAKE) test-unit
	$(MAKE) test-frontend
	$(MAKE) test-interp
	$(MAKE) test-backends
	$(MAKE) cross-validation
	$(MAKE) test-solver
	$(MAKE) test-reify
	$(MAKE) test-puzzle-sir
	$(MAKE) test-puzzle-c
