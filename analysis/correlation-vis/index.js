//////////////////////////////////////////////////////////////////////////////

var measures = ["DIbinary","DIavgall","CV","TNR","accuracy","BCR","MCC","sensitive-accuracy","0-accuracy","1-accuracy","sensitive-calibration+","sensitive-TNR","sensitive-calibration-","sensitive-TPR","TPR"];

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

function corrPlot(el, data, cScale) {
  var ps = pairs(measures, measures);
  el.append("g")
    .selectAll("rect")
    .data(ps)
    .enter()
    .append("rect")
    .attr("width", 10)
    .attr("height", 10)
    .attr("x", d => d.col * 10)
    .attr("y", d => d.row * 10)
    .attr("fill", d => cScale(correlation(data, d.v1, d.v2)))
    .attr("stroke", "none");
}

d3.csv("all_measures_numerical-binsensitive.csv", function(error, data) {
  data.forEach(d => {
    measures.forEach(m => { d[m] = Number(d[m]); });
  });

  var cScale = d3.scaleLinear().domain([-1, 0, 1]).range(
    [d3.lab(30, 80, 50),
     d3.lab(90, 0, 0),
     d3.lab(30, -80, -50)]
  );
  
  corrPlot(svg.append("g").attr("transform", "translate(840, 30)"),
           data, cScale);
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
      corrPlot(d3.select(this), data.filter(d2 => d2.algorithm === d.v1 && d2.name === d.v2), cScale);
    });

  corrPlotMatrixG.append("g")
    .selectAll("g")
    .data(algorithms)
    .enter()
    .append("g")
    .attr("transform", (d,i) => "translate(" + (5 * 160 + 20) + ", " + (190 + i * 160) + ")")
    .each(function(d) {
      corrPlot(d3.select(this), data.filter(d2 => d2.algorithm === d), cScale);
    });

  corrPlotMatrixG.append("g")
    .selectAll("g")
    .data(names)
    .enter()
    .append("g")
    .attr("transform", (d,i) => "translate(" + (i * 160) + ", " + 10 + ")")
    .each(function(d) {
      corrPlot(d3.select(this), data.filter(d2 => d2.name === d), cScale);
    });

  console.log(algorithms);
  console.log(names);
});
