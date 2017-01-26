const { ReLU, Linear, MSE, SGD, Sequential } = require('weblearn');
const ndarray = require('ndarray');
const argmax = require('compute-argmax');

global.Ai = function () {
  gamma = 0.5;

  this.init = function() {
    this.model = Sequential({
      optimizer: SGD(.01),
      loss: MSE()
    });
    this.model.add(Linear(16, 20))
              .add(ReLU())
              .add(Linear(20, 4));
    this.restart();
  }

  this.restart = function() {
    this.previousScore = 0;
    this.move = undefined;
  }

  flatten = function(grid) {
    vals = [].concat.apply([], grid.cells);
    return ndarray(vals.map(function (val) {
      if (val === null) {
        return 0;
      } else {
        return val.value;
      }
    }));
  }

  this.step = function(grid, score) {
    // The -1 is a hack to penalize trying to make illegal moves.
    // TODO: Automatically restart after game is over.
    // TODO: Actually learn strategy (currently it just eventually makes a legal move).
    reward = score - this.previousScore - 1;
    this.previousScore = score;
    previousInput = this.input;
    this.input = flatten(grid);

    if (this.move != undefined) {
      console.log(reward);
      targetQ = reward + gamma * Math.max(...this.model.forward(this.input).data);
      this.estimatedQ[this.move] = targetQ;
      this.model.fit([[previousInput, ndarray(this.estimatedQ)]]);
    }

    this.estimatedQ = this.model.forward(this.input).data;
    console.log(this.estimatedQ);
    this.move = argmax(this.estimatedQ)[0];
    return this.move;
  }
}
