//////////////////////////////////////////////////////////////////////////////

var FILENAME = "all_sampled_0.5class_0.5priv.csv";
//var FILENAME = "all_measures_numerical-binsensitive.csv";

var measures = [
  "DIbinary",
  "DIavgall",
  "CV",
  "comparative-sensitive-TPR",
  "accuracy",
  "0-accuracy",
  "1-accuracy",
  "sensitive-accuracy",
  "TNR",
  "sensitive-TNR",
  "BCR",
  "sensitive-calibration+",
  "comparative-sensitive-accuracy",
  "comparative-sensitive-TNR",
  "TPR",
  "sensitive-TPR",
  "sensitive-calibration-",

//  "percent_pos_class",
//  "MCC",
//  "comparative-sensitive-calibration+",
//  "comparative-sensitive-calibration-",
];

function pairs(l1, l2) {
  var result = [];
  l1.forEach((e1, i) => {
    l2.forEach((e2, j) => {
      result.push({ v1: e1, v2: e2, row: i, col: j });
    });
  });
  return result;
}

function uniq(l, accessor) {
  accessor = accessor || (d => d);
  var obj = {};
  var result = [];
  l.forEach(el => {
    el = accessor(el);
    if (!obj[el]) {
      obj[el] = true;
      result.push(el);
    }
  });
  return result;
}

//////////////////////////////////////////////////////////////////////////////

var svg = d3.select("#main")
    .append("svg")
    .attr("width", 1400)
    .attr("height", 4000);

var svg2 = d3.select("#main2")
    .append("svg")
    .attr("width", 350)
    .attr("height", 400);

//////////////////////////////////////////////////////////////////////////////

// "And there was a disturbance in the force", but for numerical
// analysts

function expectation(data, f) {
  var x = 0;
  data.forEach(v => { x += f(v); });
  return x / data.length;
}

function covariance(data, v1, v2) {
  return expectation(data, d => d[v1] * d[v2]) -
    expectation(data, d => d[v1]) * expectation(data, d => d[v2]);
}

function variance(data) {
  return covariance(data, data);
}

function correlation(data, c1, c2) {
  return covariance(data, c1, c2) / Math.sqrt(covariance(data, c1, c1) *
                                              covariance(data, c2, c2));
}

//////////////////////////////////////////////////////////////////////////////

var scatterplotG;
var scatterplotM1Axis, scatterplotM1AxisG, scatterplotM1AxisLabel;
var scatterplotM2Axis, scatterplotM2AxisG, scatterplotM2AxisLabel;

function scatterplot(el, data, m1, m2) {
  var update = el.selectAll("circle")
      .data(data);
  var enter = update.enter()
      .append("circle");
  var exit = update.exit()
      .remove();
  var xExt = d3.extent(data, d => d[m1]);
  var yExt = d3.extent(data, d => d[m2]);
  var xScale = d3.scaleLinear().domain(xExt).range([10, 290]);
  var yScale = d3.scaleLinear().domain(yExt).range([290, 10]);
  var cScale = d3.scaleOrdinal().range(d3.schemeCategory20);
  //var cScale = d3.scaleLinear().domain([0.3, 0.5, 0.7]).range(["white","blue", "white"]);
  
  function setAttrs(sel) {
    sel.attr("cx", d => xScale(d[m1]))
      .attr("cy", d => yScale(d[m2]))
      .attr("r", 2)
      .attr("fill", d => cScale(d.algorithm));
      //.attr("fill", d => cScale(d.percent_pos_class));
  }

  scatterplotM1Axis.scale(xScale);
  scatterplotM2Axis.scale(yScale);
  scatterplotM1AxisLabel.text(m1);

  scatterplotM1AxisG.call(scatterplotM1Axis);
  scatterplotM2AxisG.call(scatterplotM2Axis);
  scatterplotM2AxisLabel.text(m2);

  setAttrs(update);
  setAttrs(enter);
}

function corrPlot(el, data, cScale, algorithm, name) {
  var rectSize = ~~(150 / measures.length);
  var ps = pairs(measures, measures);
  el.append("g")
    .selectAll("rect")
    .data(ps)
    .enter()
    .append("rect")
    .attr("width", rectSize)
    .attr("height", rectSize)
    .attr("x", d => d.col * rectSize)
    .attr("y", d => d.row * rectSize)
    .attr("fill", d => cScale(correlation(data, d.v1, d.v2)))
    .attr("stroke", "none")
    .attr("cursor", "pointer")
    .on("click", d => {
      scatterplot(scatterplotG, data, d.v1, d.v2, algorithm, name);
    });
}

d3.csv(FILENAME, function(error, data) {
  data.forEach(d => {
    measures.forEach(m => { d[m] = Number(d[m]); });
  });
  var isFair = {
    "Calders": true,
    "Kamishima": true,
    "Kamishima-accuracy": true,
    "Kamishima-DIavgall": true,
    "ZafarBaseline": true,
    "ZafarFairness": true,
    "ZafarAccuracy": true,
    "ZafarFairness-DIavgall": true,
    "Feldman-LR": true,
    "Feldman-GaussianNB": true,
    "Feldman-SVM": true,
    "Feldman-SVM-accuracy": true,
    "Feldman-SVM-DIavgall": true,
    "Feldman-DecisionTree": true,
    "Feldman-GaussianNB-DIavgall": true,
    "Feldman-GaussianNB-accuracy": true,
	
};
  // data = data.filter(d => isFair[d.algorithm]);

  var cScale = d3.scaleLinear().domain([-1, 0, 1]).range(
    [d3.lab(30, 80, 50),
     d3.lab(90, 0, 0),
     d3.lab(30, -80, -50)]
  );

  var scatterplotMainG = svg2.append("g");
  scatterplotG = scatterplotMainG.append("g").attr("transform", "translate(20, 20)");
  scatterplotM1Axis = d3.axisBottom();
  scatterplotM1Axis.scale(d3.scaleLinear());
  scatterplotM1AxisG = scatterplotG.append("g").attr("transform", "translate(0, 290)");
  scatterplotM2Axis = d3.axisLeft();
  scatterplotM2Axis.scale(d3.scaleLinear());
  scatterplotM2AxisG = scatterplotG.append("g").attr("transform", "translate(10, 0)");

  scatterplotM1AxisLabel = scatterplotMainG.append("g").attr("transform", "translate(290, 350)")
    .append("text").style("text-anchor", "end");
  scatterplotM2AxisLabel = scatterplotMainG.append("g").attr("transform", "translate(20, 20)")
    .append("text");
  
  corrPlot(svg.append("g").attr("transform", "translate(840, 30)"),
           data, cScale, "ALL", "ALL");
  svg.append("text").attr("x", 840).attr("y", 20).text("ALL");
  svg.append("text").attr("x", 1000).attr("y", 40).text("ALL");

  var algorithms = uniq(data, d => d.algorithm);
  var names = uniq(data, d => d.name);

  var corrPlotMatrixG = svg.append("g");
  corrPlotMatrixG.attr("transform", "translate(20, 20)");
 
  corrPlotMatrixG.append("g")
    .selectAll("text")
    .data(names)
    .enter()
    .append("text")
    .attr("x", (d, i) => i * 160)
    .attr("fill", "black")
    .text(d => d);

  corrPlotMatrixG.append("g")
    .selectAll("text")
    .data(algorithms)
    .enter()
    .append("text")
    .attr("x", (160 * 6) + 20)
    .attr("y", (d, i) => (i + 1) * 160 + 40)
    .attr("fill", "black")
    .text(d => d);

  corrPlotMatrixG.append("g")
    .selectAll("g")
    .data(pairs(algorithms, names))
    .enter()
    .append("g")
    .attr("transform", d => "translate(" + (d.col * 160) + ", " + (160 + 30 + d.row * 160) + ")")
    .each(function(d) {
      corrPlot(d3.select(this),
               data.filter(d2 => d2.algorithm === d.v1 && d2.name === d.v2), cScale,
               d.v1, d.v2);
    });

  corrPlotMatrixG.append("g")
    .selectAll("g")
    .data(algorithms)
    .enter()
    .append("g")
    .attr("transform", (d,i) => "translate(" + (5 * 160 + 20) + ", " + (190 + i * 160) + ")")
    .each(function(d) {
      corrPlot(d3.select(this), data.filter(d2 => d2.algorithm === d), cScale,
               d, "ALL");
    });

  corrPlotMatrixG.append("g")
    .selectAll("g")
    .data(names)
    .enter()
    .append("g")
    .attr("transform", (d,i) => "translate(" + (i * 160) + ", " + 10 + ")")
    .each(function(d) {
      corrPlot(d3.select(this), data.filter(d2 => d2.name === d), cScale,
               "ALL", d);
    });

  console.log(algorithms);
  console.log(names);
});
