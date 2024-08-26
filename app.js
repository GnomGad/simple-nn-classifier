const fs = require("fs");
const { loadTrainingData, sigmoid, applyThreshold, loadModel } = require("./lib");

function forwardPropagationWithModel(inputs, model) {
    const [input1, input2] = inputs;

    const hiddenOutput1 = sigmoid(
        input1 * model.hiddenWeights1[0] + input2 * model.hiddenWeights1[1] + model.hiddenBias1
    );
    const hiddenOutput2 = sigmoid(
        input1 * model.hiddenWeights2[0] + input2 * model.hiddenWeights2[1] + model.hiddenBias2
    );

    const output = sigmoid(
        hiddenOutput1 * model.outputWeights[0] + hiddenOutput2 * model.outputWeights[1] + model.outputBias
    );
    return output;
}

function printTable(name, checkData, model) {
    console.log(`\n${name} Predictions:`);
    console.log("Input\tExpected\tPredicted");
    console.log("-".repeat(50));
    checkData.forEach((data) => {
        let output = forwardPropagationWithModel(data.input, model);
        let thresholdedOutput = applyThreshold(output);
        console.log(`${data.input[0]} ${name} ${data.input[1]}\t\t${data.output}\t${thresholdedOutput}`);
    });
}

const xorModel = loadModel("model_xor.json");
const andModel = loadModel("model_and.json");

const trainingData = loadTrainingData("training-data.json");

printTable("XOR", trainingData.XOR, xorModel);
printTable("AND", trainingData.AND, andModel);
