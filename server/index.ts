import tf from "@tensorflow/tfjs-node";

const maxLength = 10;
const alphaLength = 27;

const model = tf.sequential();

model.add(tf.layers.lstm({
  units: alphaLength * 2,
  inputShape: [maxLength, alphaLength],
  dropout: 0.2,
  recurrentDropout:0.2,
  useBias: true,
  returnSequences:true,
  activation:"relu"
}));

model.add(tf.layers.timeDistributed({
  layer: tf.layers.dense({
    units: alphaLength,
    activation:"softmax"
  }),
}));


model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['mse'] 
})

