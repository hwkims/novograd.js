JavaScript로 `NovoGrad` 옵티마이저를 구현하는 것은 Python의 PyTorch와 같은 고수준 라이브러리가 없기 때문에, 기초적인 텐서 연산과 메모리 관리에 대한 구현을 직접 해야 합니다. 그러나, JavaScript에서도 딥러닝 모델을 구현할 수 있는 `TensorFlow.js`와 같은 라이브러리를 사용할 수 있습니다.

여기서는 `TensorFlow.js`를 사용하여 JavaScript로 `NovoGrad` 옵티마이저를 구현하는 방법을 설명하겠습니다. `TensorFlow.js`는 딥러닝을 위한 고수준 API를 제공하며, 텐서 연산을 지원합니다.

먼저, `TensorFlow.js`를 설치하고, 그 안에서 `NovoGrad` 옵티마이저를 구현하는 코드 예시를 보겠습니다.

### 1. TensorFlow.js 설치
```bash
npm install @tensorflow/tfjs
```

### 2. NovoGrad 옵티마이저 구현 (JavaScript 코드 예시)

```javascript
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
```

### 코드 설명:
1. **NovoGrad 클래스**:
   - `params`: 옵티마이저가 최적화할 텐서 파라미터 목록입니다.
   - `lr`: 학습률.
   - `betas`: 모멘텀을 위한 두 파라미터 (`beta1`, `beta2`).
   - `eps`: 작은 값으로, 0으로 나누는 것을 방지하기 위해 사용됩니다.
   - `weightDecay`: 가중치 감소(정규화) 항목.
   - `gradAveraging`: 그래디언트 평균화를 사용할지 여부.
   - `momentumInitialized`: 모멘텀이 초기화되었는지 여부를 나타내는 플래그입니다.
   - `states`: 각 파라미터에 대한 상태를 저장하는 맵입니다. 각 파라미터마다 `v`, `m`, `gradEMA`, `step`을 저장합니다.

2. **step() 메서드**:
   - `step()` 함수는 매 학습 단계마다 호출되며, 파라미터를 업데이트하는 핵심 부분입니다.
   - 처음 실행될 때 각 파라미터에 대한 상태(`v`, `m`, `gradEMA`)를 초기화하고, `step`을 0으로 설정합니다.
   - 그 후, 그래디언트 및 관련 값들을 계산하여 파라미터를 업데이트합니다.

3. **파라미터 업데이트**:
   - `g2`는 그래디언트의 L2 노름의 제곱입니다.
   - `gradEMA`는 지수이동평균을 사용하여 계산된 그래디언트의 평활화 값입니다.
   - `m`은 그래디언트를 `v`로 나누어 스케일링한 후, 가중치 감소를 추가하여 업데이트됩니다.
   - 학습률은 `beta1`, `beta2`에 대한 바이어스 보정 후 계산됩니다.
   - 각 파라미터는 `m`을 사용하여 업데이트됩니다.

### 3. 사용 예시
```javascript
const tf = require('@tensorflow/tfjs');
const NovoGrad = require('./NovoGrad');  // NovoGrad를 정의한 파일 경로

// 예시로 임의의 파라미터와 그라디언트를 설정합니다.
const param1 = { data: tf.tensor([1.0, 2.0, 3.0]), grad: tf.tensor([0.1, 0.2, 0.3]) };
const param2 = { data: tf.tensor([4.0, 5.0, 6.0]), grad: tf.tensor([0.4, 0.5, 0.6]) };

// NovoGrad 옵티마이저 생성
const optimizer = new NovoGrad([param1, param2], { lr: 0.1 });

// 학습을 10번 반복
for (let i = 0; i < 10; i++) {
  console.log(`Step ${i + 1}`);
  optimizer.step();
  console.log(`Param1 data: ${param1.data.arraySync()}`);
  console.log(`Param2 data: ${param2.data.arraySync()}`);
}
```

### 설명:
- 이 예시에서는 두 개의 텐서 파라미터 (`param1`, `param2`)를 정의하고 그라디언트를 설정하여 옵티마이저를 테스트합니다.
- `optimizer.step()`을 호출하여 각 학습 단계마다 파라미터를 업데이트합니다.

### 요약:
- JavaScript에서 `NovoGrad` 옵티마이저를 구현하기 위해 `TensorFlow.js`를 사용하여 텐서 연산을 처리하고, PyTorch 코드에서 했던 옵티마이저의 동작을 JavaScript로 재현했습니다.
- 이 코드는 기초적인 형태의 옵티마이저로, 더 복잡한 설정이나 추가적인 기능을 위해 확장할 수 있습니다.

# PyTorch implementation of NovoGrad 

## Install 

```
pip install novograd
```

## Notice

When using NovoGrad, learning rate scheduler play an important role.  Do not forget to set it.

## Performance

### MNIST

Under Trained 3 epochs, same Architecture Neural Netwrok. 

|                | Test Acc(%) |  lr    | lr scheduler   | beta1  | beta2 | weight decay |
|:---------------|:------------|:-------|:---------------|:-------|:------|:-------------|
| Momentum SGD   |  96.92      | 0.01   | None           |  0.9   | N/A   |   0.001      |
| Adam           |  96.72      | 0.001  | None           |  0.9   | 0.999 |   0.001      |
| AdamW          |  97.34      | 0.001  | None           |  0.9   | 0.999 |   0.001      |
| NovoGrad       |  97.55      | 0.01   | cosine         |  0.95  | 0.98  |   0.001      |

## Refference
Boris Ginsburg, Patrice Castonguay, Oleksii Hrinchuk, Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, Huyen Nguyen, Jonathan M. Cohen, Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks, 	arXiv:1905.11286 [cs.LG], https://arxiv.org/pdf/1905.11286.pdf

