var Observable = require("data/observable").Observable;

function createViewModel() {
    var viewModel = new Observable();
    viewModel.scatterSource = [];
    viewModel.predictionBefore = [];
    viewModel.predictionAfter = [];
    viewModel.trueCoefficients = '';
    viewModel.randomCoefficients = '';
    viewModel.finalCoefficients = '';

    return viewModel;
}

exports.createViewModel = createViewModel;