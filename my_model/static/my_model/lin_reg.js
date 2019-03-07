let x_vals = [];
let y_vals = [];

let m, b;
let optimizerSelect;
let lr = 0.05;

const findloss = (pred, label) => pred.sub(label).square().mean();

function setup(){

    // background(100, 100, 100);
    let canvas = createCanvas(600, 400);

    canvas.mouseClicked(() => {
        if (mouseButton == LEFT){
            let x = map(mouseX, 0, width, -1, 1);
            let y = map(mouseY, 0, height, 1, -1);
            x_vals.push(x);
            y_vals.push(y);
        }
    });

    slider = createSlider(0.001, 1, 0.3, 0.001); // min, max, initial, step-value
    slider.style('width', '150px');
    slider.style('display', 'block');
    slider.changed(resetCanvas);

    optimizerSelect = createSelect();
    optimizerSelect.option('adam');
    optimizerSelect.option('sgd');
    optimizerSelect.option('momentum');
    optimizerSelect.option('adagrad');
    optimizerSelect.option('rmsprop');
    optimizerSelect.changed(resetCanvas);

    let stop = createButton('Stop');
    stop.mouseClicked(() => {
        noLoop();
    });

    let clear = createButton('Reset');
    clear.mouseClicked(() => {
        resetCanvas();
    });

    resetCanvas();
    // m = tf.variable(tf.scalar(random(1)));
    // b = tf.variable(tf.scalar(random(1)));
}


function draw(){

    background(0);

    lr = slider.value();
    fill(255).strokeWeight(0).textSize(15);
    // text(`learning rate = ${lr}`, 20, 20);
    // text(`y = ${m.dataSync()}x + ${b.dataSync()}`, 20, 35);

    tf.tidy(() => {
        if(x_vals.length > 0){

            const y = tf.tensor1d(y_vals);

            optimizer.minimize(() => {
                loss = findloss(tf.tensor1d(x_vals).mul(m).add(b), y);
                loss.data().then((mse) => {
                    fill(255).strokeWeight(0).textSize(15);
                    text(`Mean squared error = ${mse}`, 20, 50);
                });
                return loss;
            });

            const lineX = [-1, 1];

            const ys = tf.tensor1d(lineX).mul(m).add(b);

            let lineY = ys.data().then((y) => {
                let x1 = map(lineX[0], -1, 1, 0, width);
                let x2 = map(lineX[1], -1, 1, 0, width);
                let y1 = map(y[0], -1, 1, height, 0);
                let y2 = map(y[1], -1, 1, height, 0);

                // stroke(139, 0, 139);
                strokeWeight(2);
                line(x1, y1, x2, y2);
            });
        }
    });

      strokeWeight(8);
      stroke(250);
      for(let i = 0; i < x_vals.length; i++){
          let px = map(X[i], -1, 1, 0, width);
          let py = map(Y[i], -1, 1, height, 0);
          point(px, py);
      }
}

function resetCanvas(){
    X = [];
    Y = [];

    optimizer = tf.tidy(() => {
        m = tf.variable(tf.scalar(0));
        b = tf.variable(tf.scalar(0));

        const optimizers = {
            'adam' : tf.train.adam(lr),
            'sgd' : tf.train.sgd(lr),
            'momentum' : tf.train.momentum(lr, 2),
            'adagrad' : tf.train.adagrad(lr),
            'rms' : tf.train.rmsprop(lr)
        }

        return optimizers[optimizerSelect.value()];
    });

    let data = document.getElementById('data');
    data.innerText = null;

    loop();
}

function windowResized(){
    resizeCanvas(600, 400);
}
