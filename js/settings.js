function getSearchParameters() {
      var prmstr = window.location.search.substr(1);
      return prmstr != null && prmstr != "" ? transformToAssocArray(prmstr) : {};
}

function transformToAssocArray( prmstr ) {
    var params = {};
    var prmarr = prmstr.split("&");
    for ( var i = 0; i < prmarr.length; i++) {
        var tmparr = prmarr[i].split("=");
        params[tmparr[0]] = tmparr[1];
    }
    return params;
}

function go() {
    var script = $('#script').val();
    var seed = $('#seed').val();

    window.location.href = '?script=' + script + '&seed=' + seed;
}

$(document).ready(function() {
    $("#gobutton").click(go);
    var params = getSearchParameters();
    var script = params['script'];
    var seed = params['seed'];

    if (!script) {
        script = 'http://murgo.github.io/2048/js/ai_example.js';
    }
    if (!seed) {
        seed = new Date().getTime();
    }

    Math.seedrandom(seed);

    $('#script').val(script);
    $('#seed').val(seed);

    var scriptel = document.createElement('script');
    scriptel.src = script;
    //document.head.appendChild(scriptel);
});
