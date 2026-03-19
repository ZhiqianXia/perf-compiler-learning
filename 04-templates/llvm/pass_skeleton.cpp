#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class ExamplePass : public PassInfoMixin<ExamplePass> {
public:
    PreservedAnalyses run(Function& function, FunctionAnalysisManager&) {
        errs() << "ExamplePass visiting: " << function.getName() << "\n";
        return PreservedAnalyses::all();
    }
};
}  // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "ExamplePass", "0.1",
            [](PassBuilder& pass_builder) {
                pass_builder.registerPipelineParsingCallback(
                    [](StringRef name, FunctionPassManager& function_pass_manager,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (name == "example-pass") {
                            function_pass_manager.addPass(ExamplePass());
                            return true;
                        }
                        return false;
                    });
            }};
}
