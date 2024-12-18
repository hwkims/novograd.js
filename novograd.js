const tf = require('@tensorflow/tfjs');

class NovoGrad {
  constructor(params, { lr = 0.1, betas = [0.95, 0.98], eps = 1e-8, weightDecay = 0, gradAveraging = false } = {}) {
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.gradAveraging = gradAveraging;
    this.momentumInitialized = false;

    // Store parameters and their states
    this.params = params;
    this.states = new Map();
  }

  step() {
    if (!this.momentumInitialized) {
      this.params.forEach((param) => {
        const v = tf.norm(param.grad).square();
        const m = param.grad.div(tf.sqrt(v).add(this.eps)).add(this.weightDecay * param.data);

        this.states.set(param, { step: 0, v, m, gradEMA: null });
      });
      this.momentumInitialized = true;
    }

    this.params.forEach((param) => {
      const state = this.states.get(param);
      let { step, v, m, gradEMA } = state;

      let grad = param.grad;
      let g2 = tf.norm(grad).square();

      gradEMA = gradEMA ? gradEMA.mul(this.beta2).add(g2.mul(1 - this.beta2)) : g2;

      grad = grad.div(tf.sqrt(gradEMA).add(this.eps));

      if (this.gradAveraging) {
        grad = grad.mul(1 - this.beta1);
      }

      g2 = tf.norm(grad).square();
      v = v.mul(this.beta2).add(g2.mul(1 - this.beta2));
      m = m.mul(this.beta1).add(grad.div(tf.sqrt(v).add(this.eps)).add(this.weightDecay * param.data));

      const biasCorrection1 = 1 - Math.pow(this.beta1, step + 1);
      const biasCorrection2 = 1 - Math.pow(this.beta2, step + 1);

      const stepSize = this.lr * Math.sqrt(biasCorrection2) / biasCorrection1;

      param.data = param.data.sub(m.mul(stepSize));

      state.step += 1;
      state.v = v;
      state.m = m;
      state.gradEMA = gradEMA;
    });
  }
}

module.exports = NovoGrad;
