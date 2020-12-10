//===- RewriteInsertsPass.cpp - MLIR conversion pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to rewrite sequential chains of
// `spirv::CompositeInsert` operations into `spirv::CompositeConstruct`
// operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

/// Replaces sequential chains of `spirv::CompositeInsertOp` operation into
/// `spirv::CompositeConstructOp` operation if possible.
class RewriteInsertsPass
    : public SPIRVRewriteInsertsPassBase<RewriteInsertsPass> {
public:
  RewriteInsertsPass() = default;
  explicit RewriteInsertsPass(bool replacePartialInsertions)
      : replacePartialInsetions(replacePartialInsertions) {}
  void runOnOperation() override;

private:
  void replaceCompleteInsertionChains();
  void replacePartialInsertionChains();
  /// Collects a sequential insertion chain by the given
  /// `spirv::CompositeInsertOp` operation, if the given operation is the last
  /// in the chain.
  LogicalResult
  collectInsertionChain(spirv::CompositeInsertOp op,
                        SmallVectorImpl<spirv::CompositeInsertOp> &insertions);

  Operation *loadValuesEndingWithInsertions(
      Operation *op, SmallVectorImpl<Optional<Value>> &members,
      SmallVectorImpl<spirv::CompositeInsertOp> &chain);

  bool replacePartialInsetions = false;
};

} // anonymous namespace

void RewriteInsertsPass::replaceCompleteInsertionChains() {
  SmallVector<SmallVector<spirv::CompositeInsertOp, 4>, 4> workList;
  getOperation().walk([this, &workList](spirv::CompositeInsertOp op) {
    SmallVector<spirv::CompositeInsertOp, 4> insertions;
    if (succeeded(collectInsertionChain(op, insertions)))
      workList.push_back(insertions);
  });

  for (const auto &insertions : workList) {
    auto lastCompositeInsertOp = insertions.back();
    auto compositeType = lastCompositeInsertOp.getType();
    auto location = lastCompositeInsertOp.getLoc();

    SmallVector<Value, 4> operands;
    // Collect inserted objects.
    for (auto insertionOp : insertions)
      operands.push_back(insertionOp.object());

    OpBuilder builder(lastCompositeInsertOp);
    auto compositeConstructOp = builder.create<spirv::CompositeConstructOp>(
        location, compositeType, operands);

    lastCompositeInsertOp.replaceAllUsesWith(
        compositeConstructOp->getResult(0));

    // Erase ops.
    for (auto insertOp : llvm::reverse(insertions)) {
      auto *op = insertOp.getOperation();
      if (op->use_empty())
        insertOp.erase();
    }
  }
}

Operation *RewriteInsertsPass::loadValuesEndingWithInsertions(
    Operation *op, SmallVectorImpl<Optional<Value>> &members,
    SmallVectorImpl<spirv::CompositeInsertOp> &chain) {
  auto *it = op;
  while (auto insertion = dyn_cast<spirv::CompositeInsertOp>(it)) {
    chain.push_back(insertion);
    auto indicesArrayAttr = insertion.indices().cast<ArrayAttr>();
    if (indicesArrayAttr.size() == 1) {
      auto index = indicesArrayAttr[0].cast<IntegerAttr>().getInt();
      members[index] = insertion.object();
      it = insertion.composite().getDefiningOp();
    } else {
      break;
    }
  }
  return it;
}

void RewriteInsertsPass::replacePartialInsertionChains() {
  SmallVector<spirv::CompositeInsertOp, 4> insertions;
  getOperation().walk([this, &insertions](spirv::CompositeInsertOp op) {
    if (op.getType().isa<spirv::StructType>())
      insertions.push_back(op);
  });

  DenseMap<Operation *, uint32_t> numUsesMap;
  for (auto insertionOp : insertions)
    numUsesMap.insert({insertionOp.getOperation(), 0});

  for (auto insertionOp : insertions) {
    auto *parentOp = insertionOp.composite().getDefiningOp();
    if (numUsesMap.count(parentOp))
      ++numUsesMap[parentOp];
  }

  SmallVector<Operation *, 4> workList;
  for (auto insertionOp : insertions) {
    if (numUsesMap[insertionOp] == 0)
      workList.push_back(insertionOp);
  }

  while (workList.size()) {
    for (auto *op : workList) {
      SmallVector<spirv::CompositeInsertOp, 4> chain;
      SmallVector<Optional<Value>, 4> members;

      Operation *base = loadValuesEndingWithInsertions(op, members, chain);
      auto resultType = cast<spirv::CompositeInsertOp>(base)
                            .getType()
                            .cast<spirv::StructType>();
      for (uint32_t i = 0, e = resultType.getNumElements(); i < e; ++i) {
      }
    }
  }
}

LogicalResult RewriteInsertsPass::collectInsertionChain(
    spirv::CompositeInsertOp op,
    SmallVectorImpl<spirv::CompositeInsertOp> &insertions) {
  auto indicesArrayAttr = op.indices().cast<ArrayAttr>();
  // TODO: handle nested composite object.
  if (indicesArrayAttr.size() == 1) {
    auto numElements =
        op.composite().getType().cast<spirv::CompositeType>().getNumElements();

    auto index = indicesArrayAttr[0].cast<IntegerAttr>().getInt();
    // Need a last index to collect a sequential chain.
    if (index + 1 != numElements)
      return failure();

    insertions.resize(numElements);
    while (true) {
      insertions[index] = op;

      if (index == 0)
        return success();

      op = op.composite().getDefiningOp<spirv::CompositeInsertOp>();
      if (!op)
        return failure();

      --index;
      indicesArrayAttr = op.indices().cast<ArrayAttr>();
      if ((indicesArrayAttr.size() != 1) ||
          (indicesArrayAttr[0].cast<IntegerAttr>().getInt() != index))
        return failure();
    }
  }
  return failure();
}

void RewriteInsertsPass::runOnOperation() {
  replaceCompleteInsertionChains();
  if (replacePartialInsertions)
    replacePartialInsertionChains();
}

std::unique_ptr<mlir::OperationPass<spirv::ModuleOp>>
mlir::spirv::createRewriteInsertsPass(bool replacePartialInsertions) {
  return std::make_unique<RewriteInsertsPass>(replacePartialInsertions);
}
