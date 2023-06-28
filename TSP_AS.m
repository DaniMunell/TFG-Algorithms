
%==========================================================================
%
% TRABAJO FIN DE GRADO - Ingeniería Matemática UCM
%
% Daniel Munell Blanco
%
%
%                       ==========
%                       ANT SYSTEM
%                       ==========
%
%
% El siguiente código está estructurado de forma que cada sección se puede
% ejecutar independientemente del resto sin necesidad de modificar ningún
% parámetro. El código se divide en los siguientes apartados:
%
%   TEST 1 : Visualización del Rastro de Feromonas 
%
%   TEST 2 : Convergencia al Óptimo
%
%   COMPARACIÓN ALGORITMOS : ABC vs AS vs ACS (común a los tres códigos)
%
%   IMPLEMENTACIÓN DEL ALGORITMO AS
%
%   FUNCIONES AUXILIARES
%
% Los tres primeros apartados corresponden con lo visto en 
% el capítulo Experiencias Computacionales.
%
%
% NOTA : La comparación de algoritmos lleva 9 minutos por instancia.
%
% ADVERTENCIA : Los archivos de texto de cada instancia deben SOLO 
%               contener las coordenadas de las ciudades dadas en el
%               formato usado por TSPLIB. Es decir, tres columnas:
%                índice de la ciudad, coordenada x, coordenada y
%
%==========================================================================

clear all

%%

% ============================================================
%                           TEST 1
%
%             Visualización del Rastro de Feromonas       
%
% ============================================================


%           _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


% Número de ciudades
numCities = 52;

% Tomamos las coordenadas del archivo y calculamos la matriz de distancias
[coords, distMatrix] = getCoords(numCities, 'berlin52.tsp');

% Fijamos la longitud óptima de la instancia
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(1)

numAnts = numCities;   % Numero de hormigas
maxIters = 1000;       % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 5;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas
maxTime = Inf;         % Tiempo máximo permitido

listIters = [1, 100, maxIters];

% ------------------------------------------

tiledlayout(1,3)
for i = 1:length(listIters)
    maxIters = listIters(i);
      
    [~, ~, pheromones] = AS(distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
    nexttile
    plotPheromones(coords, pheromones, maxIters)
end
hold off


%%

% ============================================================
%                           TEST 2
%
%                   Convergencia al Óptimo
%
% ============================================================


%           _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


numCities = 52;
[~, distMatrix] = getCoords(numCities, 'berlin52.tsp');
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(2)

numAnts = numCities;   % Numero de hormigas
maxIters = 500;        % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 5;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas
maxTime = Inf;         % Tiempo máximo permitido

% ------------------------------------------

[~, ~, ~, bestLengthIter] = AS(distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);

plot(bestLengthIter)
title("Algoritmo AS")
xlabel("Iteraciones") 
ylabel("Mejor Longitud por Iteración")
yline(opt,'--','Óptimo','LabelHorizontalAlignment', 'left','LabelVerticalAlignment','top')
hold off


%%

% ============================================================
%                 COMPARACIÓN ALGORITMOS
%
%                    ABC vs AS vs ACS
%
% ============================================================


%               EJECUTARLO PRIMERO               < ----------------


% Número de instancias a resolver
numInstances = 4;   

% Posibles valores del parámetro maxTime
listMaxTime = [1, 5, 10, 30];
% Número de posibles valores de maxTime
numTimes = length(listMaxTime); 

% Número de ejecuciones a realizar para cada valor de maxTime
numTests = 10; 

% Vector para almacenar la mejor longitud encontrada en numTests
% ejecuciones, la media de las mejores longitudes y el error relativo
% de la media.
dataLengthAS = zeros(numTimes, 3, numInstances); 

% NOTA : 9 minutos por instancia


%%

%             _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


numCities = 52;
[~, distMatrix] = getCoords(numCities, 'berlin52.tsp');
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(10)

numAnts = numCities;   % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 5;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
    dataLengthAS(i,:,1) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |  PROBLEMA : kroA100 |
%          |_____________________|


numCities = 100;
[coords, distMatrix] = getCoords(numCities, 'kroA100.tsp');
opt = 21285.44;


% --------------- PARÁMETROS ---------------

rng(10)

numAnts = numCities;   % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 3;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
    dataLengthAS(i,:,2) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |   PROBLEMA : d198   |
%          |_____________________|


numCities = 198;
[~, distMatrix] = getCoords(numCities, 'd198.tsp');
opt = 15780;


% --------------- PARÁMETROS ---------------

rng(10)

numAnts = numCities;   % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 3;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
    dataLengthAS(i,:,3) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |  PROBLEMA : lin318  |
%          |_____________________|


numCities = 318;
[~, distMatrix] = getCoords(numCities, 'lin318.tsp');
opt = 42029;


% --------------- PARÁMETROS ---------------

rng(10)

numAnts = numCities;   % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
alpha = 1;             % Factor importancia feromonas
beta = 4;              % Factor importancia visibilidad
rho = 0.5;             % Tasa de evaporacion
Q = 1;                 % Cantidad base deposito de feromonas

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
    dataLengthAS(i,:,4) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

% ============================================================
%                         ALGORITMO
% ============================================================



function [bestLength, bestSolution, pheromones, bestLengthIter] = AS(distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime)
% =======================================================
% Aplica el algoritmo ABC (Ant System)
%   
%   INPUTS:
%       
%       distMatrix : matriz de distancias entre ciudades
%       alpha : factor de importancia de las feromonas
%       beta : factor de importancia de la heurística (visibilidad)
%       rho : tasa de evaporación de las feromonas
%       numAnts : número de hormigas empleadas
%       maxIters : máximo número de iteraciones permitidas
%       Q : factor multiplicativo deposito de feromonas
%       maxTime : máximo tiempo de ejecución 
%
%   OUTPUTS:
%
%       bestLength : longitud del mejor tour encontrado
%       bestSolution : permutación del mejor tour encontrado
%
% =======================================================


    % -------------------- INICIALIZACIÓN --------------------

    numCities = size(distMatrix, 1);    % Numero de ciudades
    
    % Elección del valor inicial de las feromonas
    [nnTourLength, ~] = NN(distMatrix);
    initialPheromone = numAnts*Q/nnTourLength;
    
    % Inicialización matriz de feromonas y visibilidad
    pheromones = initialPheromone * ones(numCities, numCities);
    visibility = (1./distMatrix) .^ beta;
    probs = zeros(1, numCities); 
    
    % Inicialización de la mejor solución inicial
    bestSolution = zeros(1, numCities);
    bestLength = Inf;
    
    bestLengthIter = zeros(1, maxIters);

    % --------------------------------------------------------
    
    startTime = tic;

    % Bucle Principal
    for iter = 1:maxIters
        
        if toc(startTime) > maxTime
            return
        end

        % Inicialización de los tours y su longitud
        antPaths = zeros(numAnts, numCities);
        antPathsLength = zeros(1, numAnts);
        
        % Se calcula el valor del numerador de la formula que
        % da la probabilidad de elección de cada tramo
        % (se usa la función máximo para evitar que el valor se
        % haga 0 cuando se acerca al menor valor máquina permitido)
        probMatrix = max((pheromones .^ alpha) .* visibility, eps(0));



        % ------------------ CONSTRUCCIÓN DE SOLUCIONES -------------------
        
        % Se constuye el tour para cada hormiga secuencialmente
        for ant = 1:numAnts

            % Se elige una ciudad inicial aleatoria
            currentCity = randi(numCities);
            antPaths(ant, 1) = currentCity; 

            % Se inicializa la lista de ciudades no visitadas
            allowedCities = true(1, numCities);
            allowedCities(currentCity) = false;
                     
            for cityIndex = 2:numCities
    
                % Se calcula la probabilidad de elegir cada ciudad
                probsSum = 0;
                for l = 1:numCities
                    if allowedCities(l)
                        probNumerator = probMatrix(currentCity, l);
                        
                        probs(l) = probNumerator;
                        probsSum = probsSum + probNumerator;
                    else
                        probs(l) = 0;
                    end
                end
                realProbs = probs / probsSum;
                
                % Se elige la siguiente ciudad en base a esta probabilidad
                newCity = chooseCity(realProbs);
                
                % Se actualiza el tour y su longitud
                antPaths(ant, cityIndex) = newCity;
                antPathsLength(ant) = antPathsLength(ant) + distMatrix(currentCity, newCity);
                
                % Se actualiza la ciudad actual
                allowedCities(newCity) = false;
                currentCity = newCity;
            end
            firstCity = antPaths(ant, 1);
            antPathsLength(ant) = antPathsLength(ant) + distMatrix(currentCity, firstCity);
            
        end
        

        % Se busca la mejor solución hasta el momento
        [minLength, minIndex] = min(antPathsLength);
        if minLength < bestLength
            bestLength = minLength;
            bestSolution = antPaths(minIndex, :);
        end

        bestLengthIter(iter) = minLength;

        % -----------------------------------------------------------------



        % ------------------- ACTUALIZACIÓN FEROMONAS ---------------------

        % Evaporación de las feromonas
        pheromones = (1 - rho) * pheromones;
            

        % Cada hormiga deposita una cantidad de feromonas en los
        % tramos por los que ha pasado
        for ant = 1:numAnts
    
            antDelta = Q / antPathsLength(ant);      
            
            for cityIndex = 1:numCities
                i = antPaths(ant, cityIndex);
    
                if cityIndex == numCities
                    j = antPaths(ant, 1);
                else
                    j = antPaths(ant, cityIndex + 1);
                end
                
                pheromones(i, j) = pheromones(i, j) + antDelta;
                pheromones(j, i) = pheromones(i, j);
            end

        end   
       

        % -----------------------------------------------------------------

    end
    

%     % Muestra la longitud de la mejor solución encontrada
%     fprintf('---------\n')
%     fprintf('Mejor Longitud: %s\n', num2str(bestLength));


end




%%



% ============================================================
%                         FUNCIONES
% ============================================================


function [nnTourLength, nnTour] = NN(distMatrix)
% =======================================================
% Aplica la heurística del vecino más cercano para TSP
%   
%   INPUTS:
%
%       distMatrix : matriz de distancias entre ciudades
%
%   OUTPUTS:
%
%       nnTourLength : longitud del tour encontrado
%       nnTour : permutación del tour encontrado
%
% =======================================================
    numCities = size(distMatrix, 1);
    
    nnTour = zeros(1, numCities);
    nnTourLength = 0;
    allowedCities = true(1, numCities);
    
    origin = randi(numCities);
    nnTour(1) = origin;
    allowedCities(origin) = false;
    for cityIndex = 1:(numCities-1)

        currentCity = nnTour(cityIndex);
        
        minDistance = Inf;
        for city = 1:numCities
            if allowedCities(city)
                distToCity = distMatrix(currentCity, city);
                if distToCity < minDistance
                    minDistance = distToCity;
                    minIndex = city;
                end
            end
        end

        nnTour(cityIndex + 1) = minIndex;
        nnTourLength = nnTourLength + minDistance;
        allowedCities(minIndex) = false;
    end
    nnTourLength = nnTourLength + distMatrix(nnTour(numCities), nnTour(1));

end


% ------------------------------------------------------------


function selectedIndex = chooseCity(probs)
% =======================================================
% Elige una ciudad en función de las probabilidades dadas
%   
%   INPUTS:
%
%       probs : vector de probabilidades (debe sumar 1)
%
%   OUTPUTS:
%
%       selectedIndex : índice de la ciudad elegida
%
% =======================================================

    numCities = length(probs);

    r = rand();
    cumSums = cumsum(probs);
    
    for selectedIndex = 1:numCities
        if r <= cumSums(selectedIndex)
            return
        end
    end
end


% ------------------------------------------------------------


function [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime)
% =======================================================
% Ejecuta el algoritmo AS tantas veces como se indique
% y calcula la longitud de tour media obtenida
%   
%   INPUTS:
%
%       numTests : número de ejecuciones del algoritmo
%       ... : veáse parámetros de la función AS
%
%   OUTPUTS:
%       
%       allBestLength : mejor longitud encontrada
%       meanBestLength : longitud de tour media
%       meanTime : tiempo de ejecución medio
%
% =======================================================

    meanBestLength = 0;
    allBestLength = Inf;
    meanTime = 0;

    for tests = 1:numTests
        tic
        [bestLength, ~] = AS(distMatrix, alpha, beta, rho, numAnts, maxIters, Q, maxTime);
        execTime = toc;

        if bestLength < allBestLength
            allBestLength = bestLength;
        end

        meanBestLength = meanBestLength + bestLength;
        meanTime = meanTime + execTime;
    end
    meanBestLength = meanBestLength / numTests;
    meanTime = meanTime / numTests;
end


% ------------------------------------------------------------


function plotPheromones(coords, pheromones, iter)
% =======================================================
% Crea un grafo de la intensidad del rastro de feromonas,
% a mayor opacidad, mayor concentración.
%   
%   INPUTS:
%
%       coords : coordenadas de las ciudades
%       pheromones : matriz de nivel de feromonas
%       iter : iteración actual
%
%   OUTPUTS:
%       
%       p : grafo intensidad feromonas
%
% ======================================================= 

    numCities = size(coords, 1);

    minP = min(min(pheromones));
    maxP = max(max(pheromones));
    max_minP = maxP - minP;

    plot(coords(:, 1), coords(:, 2), 'bo');
    hold on

    for i = 1:numCities
        for j = i:numCities
            opacity = max( (pheromones(i,j) - minP)/max_minP, 0 );
            plot(coords([i, j], 1), coords([i, j], 2), 'b-', 'LineWidth', 3,'Color', [1,0,0,opacity])
        end
    end

    xlabel('x');
    ylabel('y');
%     title("Concentración del Rastro de Feromonas (iter = " + iter + ")");
    title("Iteración = " + iter);
    legend('Ciudades');
    hold off

end


% ------------------------------------------------------------


function [coords, distMatrix] = getCoords(numCities, fileName)
% =======================================================
% Almacena las coordenadas de un archivo dado y calcula
% la matriz de distancias
%   
%   INPUTS:
%
%       numCities : numero de ciudades del problema
%       fileName : nombre del archivo
%
%   OUTPUTS:
%       
%       coords : vector de coordenadas x,y
%       distMatrix : matriz de distancias
%
% ADVERTENCIA : Los archivos de texto de cada instancia deben SOLO 
%               contener las coordenadas de las ciudades dadas en el
%               formato usado por TSPLIB. Es decir, tres columnas:
%               índice de la ciudad, coordenada x, coordenada y
%
% ======================================================= 

    dataFile = fopen(fileName,'r');
    
    coords = fscanf(dataFile, '%d %f %f', [3 numCities])';
    coords = coords(:,2:3);
    
    distMatrix = squareform(pdist(coords));

end