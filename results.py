class PyLSpmHTML(object):

    def __init__(self, plsobject):
        self.path_matrix = plsobject.path_matrix
        self.path_matrix_low = plsobject.path_matrix_low
        self.path_matrix_high = plsobject.path_matrix_high
        self.path_matrix_range = plsobject.path_matrix_range

        self.corLVs = plsobject.corLVs()
        self.AVE = plsobject.AVE()
        self.fscores = plsobject.fscores
        self.comunalidades = plsobject.comunalidades()
        self.outer_loadings = plsobject.outer_loadings
        self.xloads = plsobject.xloads()
        self.outer_weights = plsobject.outer_weights
        self.fscores = plsobject.fscores
        self.rhoA = plsobject.rhoA()
        self.alpha = plsobject.alpha()
        self.r2 = plsobject.r2
        self.htmt = plsobject.htmt()
        self.cr = plsobject.cr()
        self.total_effects = plsobject.total_effects
        self.indirect_effects = plsobject.indirect_effects
        self.empirical = plsobject.empirical()
        self.implied = plsobject.implied()
        self.scheme = plsobject.scheme
        self.regression = plsobject.regression

        self.srmr = plsobject.srmr()
        self.frequency = plsobject.frequency()
        self.mean = plsobject.dataInfo()[0]
        self.sd = plsobject.dataInfo()[1]

    def geraInfo(self):

        print_matrix = """
        <div id=info>
        <h3>Model Info</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        linhas = ['Scheme', 'Regression', 'Latent Variables',
                  'Manifests', 'Observations', 'SRMR']
        conteudo = [(self.scheme), self.regression, len(self.path_matrix), len(
            self.outer_loadings), len(self.fscores), round(self.srmr, 3)]

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            print_matrix += "<td>" + str(conteudo[i]) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def geraTable(self, matrix, titulo, link):

        print_matrix = """
        <div id=""" + link + """>
        <h3>""" + titulo + """</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        colunas = matrix.columns.values
        linhas = matrix.index.values
        conteudo = matrix.values

        print_matrix += "<th></th>"
        for i in range(len(colunas)):
            print_matrix += "<th>" + str(colunas[i]) + "</th>"

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            for j in range(len(colunas)):
                if (str(conteudo[i][j]) == '0.0'):
                    print_matrix += "<td></td>"
                else:
                    print_matrix += "<td>" + \
                        str(round(conteudo[i][j], 3)) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def geraTableStr(self, matrix, titulo, link):

        print_matrix = """
        <div id=""" + link + """>
        <h3>""" + titulo + """</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        colunas = matrix.columns.values
        linhas = matrix.index.values
        conteudo = matrix.values

        print_matrix += "<th></th>"
        for i in range(len(colunas)):
            print_matrix += "<th>" + str(colunas[i]) + "</th>"

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            for j in range(len(colunas)):
                if (str(conteudo[i][j]) == '0.0 0.0'):
                    print_matrix += "<td></td>"
                else:
                    print_matrix += "<td>" + str(conteudo[i][j]) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def geraReliabilityTable(self, matrix, matrix2, matrix3):

        print_matrix = """
        <div id="reliability">
        <h3>Construct Reliability</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        linhas = matrix.index.values
        conteudo = matrix.values
        conteudo2 = matrix2.values
        conteudo3 = matrix3.values

        print_matrix += "<th></th>"
        print_matrix += "<th>Cronbach Alpha</th>"
        print_matrix += "<th>Composite Reliability</th>"
        print_matrix += "<th>&rho;A</th>"

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo[i]), 3)) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo2[i]), 3)) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo3[i]), 3)) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def geraDataInfoTable(self, matrix, matrix2):

        print_matrix = """
        <div id="datainfo">
        <h3>Population Info</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        linhas = matrix.index.values
        conteudo = matrix.values
        conteudo2 = matrix2.values

        print_matrix += "<th></th>"
        print_matrix += "<th>Mean</th>"
        print_matrix += "<th>Standard Deviation</th>"

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo[i]), 3)) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo2[i]), 3)) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def gerasingleTable(self, matrix, titulo, link):

        print_matrix = """
        <div id=""" + link + """>
        <h3>""" + titulo + """</h3>
            <table class="table table-striped table-condensed">
                <thead>
                    <tr>"""

        linhas = matrix.index.values
        conteudo = matrix.values

        print_matrix += "<th></th>"
        print_matrix += "<th>" + titulo + "</th>"

        print_matrix += """</tr></thead><tbody>"""

        for i in range(len(linhas)):
            print_matrix += "<tr>"
            print_matrix += "<td>" + str(linhas[i]) + "</td>"
            print_matrix += "<td>" + \
                str(round(float(conteudo[i]), 3)) + "</td>"
            print_matrix += "</tr>"

        print_matrix += """</tbody>
            </table></div>"""

        return print_matrix

    def generate(self):

        message = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>PyLS-PM - Partial Least Squares Path Modeling in Python</title>

        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

        <script
          src="https://code.jquery.com/jquery-2.2.4.min.js"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44="
          crossorigin="anonymous"></script>
        <!-- Optional theme -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">

        <!-- Latest compiled and minified JavaScript -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>

        <style>
          body {
              position: relative;
          }
          ul.nav-pills {
              top: 20px;
              margin-left: -22px;
              margin-right: -22px;
          }
          .nav-pills>li>a {
            padding: 5px 15px !important;
          }
          .sidebar {
            position: fixed;
            top: 41px;
            bottom: 0;
            left: 0;
            z-index: 1000;
            display: block;
            padding: 20px;
            overflow-x: hidden;
            overflow-y: auto; /* Scrollable contents if viewport is shorter than content. */
            background-color: #f5f5f5;
            border-right: 1px solid #eee;
          }
          .results {
            padding-top: 51px;
            padding-left: 50px;
            padding-right: 50px;
          }

          </style>

        <script>
        var offsetHeight = 51;
        $(document).ready(function(){

        $('a').click(function (event) {
        var tamanho = $('.results').find($(this).attr('href'))

            $('body,html').animate({
                scrollTop: tamanho[0].offsetTop-60
            }, 500);
            return false;
        });
        });
        </script>

        </head>"""

        info = self.geraInfo()

        path_matrix = self.geraTable(
            self.path_matrix, 'Path Coefficients', 'path_matrix')

        if(self.regression == 'fuzzy'):
            path_matrix_low = self.geraTable(
                self.path_matrix_low, 'Low Path Coefficients', 'path_matrix_low')
            path_matrix_high = self.geraTable(
                self.path_matrix_high, 'High Path Coefficients', 'path_matrix_high')
            path_matrix_range = self.geraTableStr(
                self.path_matrix_range, 'Path Coefficients Range', 'path_matrix_range')

        indirect_effects = self.geraTable(
            self.indirect_effects, 'Indirect Effects', 'indirect_effects')
        total_effects = self.geraTable(
            self.total_effects, 'Total Effects', 'total_effects')

        r2 = self.gerasingleTable(self.r2, 'R-Squared', 'r2')
        AVE = self.gerasingleTable(
            self.AVE, 'Average Variance Extracted', 'AVE')

        corLVs = self.geraTable(
            self.corLVs, 'Latent Variables Correlations', 'corLVs')
        htmt = self.geraTable(
            self.htmt, 'Heterotrait-Monotrait Ratio of Correlations (HTMT)', 'htmt')
        outer_loadings = self.geraTable(
            self.outer_loadings, 'Loadings', 'outer_loadings')
        comunalidades = self.geraTable(
            self.comunalidades, 'Communalities', 'comunalidades')
        xloads = self.geraTable(self.xloads, 'Crossloadings', 'xloads')
        fscores = self.geraTable(self.fscores, 'Scores', 'fscores')
        outer_weights = self.geraTable(
            self.outer_weights, 'Weigths', 'outer_weights')

        empirical = self.geraTable(
            self.empirical, 'Empirical Correlation Matrix', 'empirical')
        implied = self.geraTable(
            self.implied, 'Model Implied Correlation Matrix', 'implied')

        frequency = self.geraTable(
            self.frequency, 'Frequency Table', 'frequency')

        reliability = self.geraReliabilityTable(self.alpha, self.cr, self.rhoA)

        datainfo = self.geraDataInfoTable(self.mean, self.sd)

        body = """<body data-spy="scroll" data-target="#myScrollspy" data-offset="60">
        <nav class="navbar navbar-inverse navbar-fixed-top"><div class="container-fluid"><div class="navbar-header"><div class="navbar-brand">PyLS-PM</div></div></div></nav>
        <div class="container-fluid">
        <div class="row">
        <div class="col-sm-3 col-md-2 sidebar" id="myScrollspy">
        <ul class="nav nav-pills nav-stacked">
        <div align="center">
        <img src="logo.png"></div>
        <li class=""><a align="center" href="#overall"><b>Overall</b></a></li>
        <li class=""><a href="#reliability">Construct Reliability</a></li>
        <li class=""><a href="#htmt">HTMT</a></li>

        <li class=""><a align="center" href="#inner"><b>Inner Model</b></a></li>
        <li class=""><a href="#path_matrix">Path Coefficients</a></li>"""

        if(self.regression == 'fuzzy'):
            body += """
            <li class=""><a href="#path_matrix_low">Low Path Coefficients</a></li>
            <li class=""><a href="#path_matrix_high">High Path Coefficients</a></li>
            <li class=""><a href="#path_matrix_range">Path Coefficients Range</a></li>"""
        else:
            body += """<li class=""><a href="#r2">R-Squared</a></li>"""

        body += """<li class=""><a href="#indirect_effects">Indirect Effects</a></li>
        <li class=""><a href="#total_effects">Total Effects</a></li>
        <li class=""><a href="#AVE">Average Variance Extracted</a></li>
        <li class=""><a href="#corLVs">Latent Variables Correlations</a></li>

        <li class=""><a align="center" href="#outer"><b>Outer Model</b></a></li>
        <li class=""><a href="#outer_loadings">Loadings</a></li>
        <li class=""><a href="#comunalidades">Communalities</a></li>
        <li class=""><a href="#xloads">Crossloadings</a></li>
        <li class=""><a href="#outer_weights">Weigths</a></li>

        <li class=""><a align="center" href="#others"><b>Others</b></a></li>        
        <li class=""><a href="#fscores">Scores</a></li>
        <li class=""><a href="#empirical">Empirical Correlation Matrix</a></li>
        <li class=""><a href="#implied">Model Implied Correlation Matrix</a></li>
        <li class=""><a align="center" href="#datainfo"><b>Data Info</b></a></li>   
        </ul>
        </div>
        <div class="col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 results">"""

        rodape = """</div></div></div>
        </body>
        </html>"""

        f = open('results.html', 'w', encoding='utf-8')
        f.write(message)
        f.write(body)

        f.write('<h1 id="overall">Overall</h1><hr>')
        f.write(info)
        f.write(reliability)
        f.write(htmt)

        f.write('<h1 id="inner">Inner Model</h1><hr>')
        f.write(path_matrix)

        if(self.regression == 'fuzzy'):
            f.write(path_matrix_low)
            f.write(path_matrix_high)
            f.write(path_matrix_range)
        else:
            f.write(r2)
        f.write(indirect_effects)
        f.write(total_effects)
        f.write(AVE)
        f.write(corLVs)

        f.write('<h1 id="outer">Outer Model</h1><hr>')

        f.write(outer_loadings)
        f.write(comunalidades)
        f.write(xloads)
        f.write(outer_weights)

        f.write('<h1 id="others">Others</h1><hr>')
        f.write(fscores)
        f.write(empirical)
        f.write(implied)
        f.write('<h1 id="datainfo">Data Info</h1><hr>')
        f.write(frequency)
        f.write(datainfo)
        f.write(rodape)
        f.close()
