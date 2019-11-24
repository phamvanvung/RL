const N_TRIALS = 2000;
const N_ARMS = 16;
const N_FEATURES = 5;
const BEST_ARMS = [1, 3, 5];

/**
 * Simulate the learning process and observe the payoffs
 * @param alpha number: this is the :math:`\alpha = \sqrt{ln(2/\sigma)/2}`
 * @param X {Array: nTrials x nArms x nFeatures}: The observed data
 * @param generateReward {function}: A function used to generate the reward data
 * @param trueTheta {Array: nArms x nFeatures}: true theta used to generate reward
 * @return {Array}: the payoffs for every trial.
 */
function linUCB(alpha, X, generateReward, trueTheta) {
    const nTrials = X.length;
    const nArms = X[0].length;
    const nFeatures = X[0][0].length;
    //Initialize A

    let A = [];
    let b = [];
    let theta = [];
    let p = [];
    let payoffs = [];
    for (let a = 0; a < nArms; a++) {
        A.push(math.identity(nFeatures));
        b.push(math.zeros(nFeatures));
    }
    for (let t = 0; t < nTrials; t++) {
        theta[t] = [];
        p[t] = [];
        // compute the estimates (theta) and prediction for all arms
        for (let a = 0; a < nArms; a++) {
            let invA = math.inv(A[a]);
            theta[t][a] = invA._data.map(row => {
                return math.dot(row, b[a]);
            });
            let ctxDotInvA = invA._data.map(row => {
                return math.dot(row, X[t][a]);
            })
            p[t][a] = math.dot(theta[t][a], X[t][a]) + alpha * math.sqrt(math.dot(ctxDotInvA, X[t][a]));
        }
        let chosenArm = math.argmax(p[t]);
        let xChosenArm = X[t][chosenArm];
        payoffs[t] = generateReward(chosenArm, xChosenArm, trueTheta);
        // update intermediate object
        let xMat = xChosenArm.map(f => [f]);
        A[chosenArm] = math.add(A[chosenArm], math.multiply(xMat, math.transpose(xMat)));
        b[chosenArm] = math.add(b[chosenArm], xChosenArm.map(v => v * payoffs[t]));
    }
    return payoffs;
}
