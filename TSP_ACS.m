
%==========================================================================
%
% TRABAJO FIN DE GRADO - Ingeniería Matemática UCM
%
% Daniel Munell Blanco
%
%
%                       =================
%                       ANT COLONY SYSTEM
%                       =================
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
%   (En este archivo se obtienen las gráficas relativas a los resultados)
%
%   TEST 3: Una Instancia de Mayor Tamaño
%
%   IMPLEMENTACIÓN DEL ALGORITMO ACS
%
%   FUNCIONES AUXILIARES
%
% Los cuatro primeros apartados corresponden con lo visto en 
% el capítulo Experiencias Computacionales.
%
%
% NOTA : La comparación de algoritmos lleva 9 minutos por instancia.
%
% NOTA : La instancia de mayor tamaño lleva 10 minutos.
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

numAnts = 10;          % Numero de hormigas
maxIters = 1000;       % Iteraciones maximas
beta = 5;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria
maxTime = Inf;         % Tiempo máximo permitido

listIters = [1, 100, maxIters];

% ------------------------------------------

tiledlayout(1,3)
for i = 1:length(listIters)
    maxIters = listIters(i);
      
    [~, ~, pheromones] = ACS(distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
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

numAnts = 10;          % Numero de hormigas
maxIters = 500;        % Iteraciones maximas
beta = 5;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria
maxTime = Inf;         % Tiempo máximo permitido

% ------------------------------------------

[~, ~, ~, bestLengthIter] = ACS(distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);

plot(bestLengthIter)
ylim([7500, 9000])
title("Algoritmo ACS")
xlabel("Iteraciones") 
ylabel("Mejor Longitud por Iteración")
yline(opt,'--','Óptimo','LabelHorizontalAlignment', 'right','LabelVerticalAlignment','top')
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
dataLengthACS = zeros(numTimes, 3, numInstances); 

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

numAnts = 10;          % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
beta = 5;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
    dataLengthACS(i,:,1) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |  PROBLEMA : kroA100 |
%          |_____________________|


numCities = 100;
[~, distMatrix] = getCoords(numCities, 'kroA100.tsp');
opt = 21285.44;


% --------------- PARÁMETROS ---------------

rng(10)

numAnts = 10;          % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
beta = 3;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
    dataLengthACS(i,:,2) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
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

numAnts = 10;          % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
beta = 3;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
    dataLengthACS(i,:,3) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
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

numAnts = 10;          % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
beta = 4;              % Factor importancia visibilidad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria

% ------------------------------------------

for i = 1:length(listMaxTime)
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
    dataLengthACS(i,:,4) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

% EJECUTAR UNA VEZ SE HAN OBTENIDO LOS DATOS DE TODAS LAS INSTANCIAS

errorABC = squeeze(dataLengthABC(:, end,:))';
errorAS = squeeze(dataLengthAS(:, end,:))';
errorACS = squeeze(dataLengthACS(:, end,:))';

errorABC(errorABC < 10^-6) = 0;
errorAS(errorAS < 10^-6) = 0;
errorACS(errorACS < 10^-6) = 0;
 

%%

% Gráfica del ratio de mejora del error relativo con el tiempo

ratioABC = errorABC(:,1)./errorABC(:,4);
ratioAS = errorAS(:,1)./errorAS(:,4);
ratioACS = errorACS(:,1)./errorACS(:,4);

plot([ratioABC, ratioAS, ratioACS], '.-', 'MarkerSize', 24)
title("Ratio de Mejora del Error Relativo de 1s a 30s")
xlabel("Instancia")
ylabel("^{Error Relativo a 1s}/_{Error Relativo a 30s}")
legend('ABC', 'AS', 'ACS')
xticks([1, 2, 3, 4])
xticklabels({'berlin52', 'kroA100', 'd198', 'lin318'})


%%

% Gráfica del error relativo de de AS vs ACS a los 30s

bar([errorAS(:,4), errorACS(:,4)])
title("Error Relativo a los 30s")
xlabel("Instancia")
ylabel("Error Relativo (%)")
legend('AS', 'ACS', 'Location','northwest')
xticks([1, 2, 3, 4])
xticklabels({'berlin52', 'kroA100', 'd198', 'lin318'})


%%


% ============================================================
%                           TEST 3
%
%                Una Instancia de Mayor Tamaño       
%
% ============================================================


%           _____________________ 
%          |                     |
%          |  PROBLEMA : pr2392  |
%          |_____________________|


numCities = 2392;
[~, distMatrix] = getCoords(numCities, 'pr2392.tsp');
opt = 378032;


% --------------- PARÁMETROS ---------------

rng(3)

numAnts = 10;          % Numero de hormigas
maxIters = 10^6;       % Iteraciones maximas
beta = 5;              % Factor importancia visibil     idad
rho = 0.1;             % Tasa de evaporacion global
xi = 0.1;              % Tasa de evaporacion local
q0 = 0.9;              % Parametro regla pseudoaleatoria
maxTime = 600;         % Tiempo máximo permitido

% ------------------------------------------

[bestLength, bestSolution] = ACS(distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);

bestLength/opt - 1


%%

% ============================================================
%                         ALGORITMO
% ============================================================



function [bestLength, bestSolution, pheromones, bestLengthIter] = ACS(distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime)
% =======================================================
% Aplica el algoritmo ACS (Ant Colony System)
%   
%   INPUTS:
%       
%       distMatrix : matriz de distancias entre ciudades¡
%       beta : factor de importancia de la heurística (visibilidad)
%       rho : tasa de evaporación global
%       xi : tasa de evaporación local
%       q0 : parámetro regla pseudoaleatoria elección ciudad
%       numAnts : número de hormigas empleadas
%       maxIters : máximo número de iteraciones permitidas  
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
    initialPheromone = 1/(numCities*nnTourLength);
    antLocalDelta = xi*initialPheromone;
    
    % Inicialización matriz de feromonas y visibilidad
    pheromones = initialPheromone * ones(numCities, numCities);
    visibility = (1./distMatrix) .^ beta;

    probs = zeros(1, numCities); 
    probMatrix = pheromones .* visibility; % Numerador

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
        allowedCities = true(numAnts, numCities);
    

        % ------------------ CONSTRUCCIÓN DE SOLUCIONES -------------------       
        
        for ant = 1:numAnts

            % Se elige una ciudad inicial aleatoria
            currentCity = randi(numCities);
            antPaths(ant, 1) = currentCity; 

            % Se elimina la ciudad de la lista de no visitadas
            allowedCities(ant, currentCity) = false;
        end


        for cityIndex = 1:numCities
            
            % Se constuye el tour para cada hormiga paralelamente
            for ant = 1:numAnts

                if cityIndex < numCities
                    
                    % Ciudad actual de la hormiga
                    currentCity = antPaths(ant, cityIndex);
    
                    % Se elige la siguiente ciudad en base a
                    % una regla pseudoaleatoria
                    q = rand();
                       
                    if q <= q0 
                        % Se obtiene el maximo numerador entre 
                        % las ciudades no visitadas
                        maxArg = -1;
                        for l = 1:numCities
                            if allowedCities(ant, l)
                                arg = probMatrix(currentCity, l);
                                if arg > maxArg
                                    maxArg = arg;
                                    maxIndex = l;
                                end
                            end
                        end
                        
                        % Se elige la siguiente ciudad dado el indice del
                        % maximo
                        newCity = maxIndex;
        
                    else                 
                        % Se calcula la probabilidad de elegir cada ciudad
                        probsSum = 0;
                        for l = 1:numCities
                            if allowedCities(ant, l)
                                probNumerator = probMatrix(currentCity, l);
                                
                                probs(l) = probNumerator;
                                probsSum = probsSum + probNumerator;
                            else
                                probs(l) = 0;
                            end
                        end
                        realProbs = probs / probsSum;
                        
                        % Se elige la siguiente ciudad en base a esta
                        % probabilidad
                        newCity = chooseCity(realProbs);                 
                    end
    
                    % Se actualiza el tour y su longitud
                    antPaths(ant, cityIndex + 1) = newCity;
                    antPathsLength(ant) = antPathsLength(ant) + distMatrix(currentCity, newCity);
                    
                    % Se actualiza la lista de ciudades no visitadas
                    allowedCities(ant, newCity) = false;                    


                else % cityIndex == numCities
        
                    % Se añade la distancia de la ciudad final a la inicial
                    firstCity = antPaths(ant, 1); lastCity = antPaths(ant, numCities);
                    antPathsLength(ant) = antPathsLength(ant) + distMatrix(lastCity, firstCity);

                end

            end
        
        % -----------------------------------------------------------------
        


        % --------------- ACTUALIZACIÓN FEROMONAS LOCAL -------------------

            % Una vez todas las hormigas se han movido se actualiza para
            % cada una las feromonas locales del tramo que han utilizado 
            % y el valor del numerador de la regla pseudoaleatoria
            for ant = 1:numAnts    

                i = antPaths(ant, cityIndex);

                if cityIndex == numCities
                    j = antPaths(ant, 1);
                else
                    j = antPaths(ant, cityIndex + 1);
                end
                
                pheromones(i, j) = (1 - xi)*pheromones(i, j) + antLocalDelta;
                pheromones(j, i) = pheromones(i, j);

                probMatrix(i, j) = pheromones(i, j)*visibility(i, j);
                probMatrix(j, i) = probMatrix(i, j);     
    
            end  

        end 

        % -----------------------------------------------------------------



        % Se busca la mejor solución hasta el momento
        [minLength, minIndex] = min(antPathsLength);
        if minLength < bestLength
            bestLength = minLength;
            bestSolution = antPaths(minIndex, :);
        end

        bestLengthIter(iter) = minLength;


        % --------------- ACTUALIZACIÓN FEROMONAS GLOBAL ------------------

        % Solo se evaporan y depositan feromonas globales en el mejor tour
        bestAntDelta = rho / bestLength;
        
        for cityIndex = 1:numCities
            i = bestSolution(cityIndex);

            if cityIndex == numCities
                j = bestSolution(1);
            else
                j = bestSolution(cityIndex + 1);
            end
            
            pheromones(i, j) = (1 - rho)*pheromones(i, j) + bestAntDelta;
            pheromones(j, i) = pheromones(i, j);

            probMatrix(i, j) = pheromones(i, j)*visibility(i, j);
            probMatrix(j, i) = probMatrix(i, j);
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
    
    originCity = randi(numCities);
    nnTour(1) = originCity;
    allowedCities(originCity) = false;
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


function [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime)
% =======================================================
% Ejecuta el algoritmo AS tantas veces como se indique
% y calcula la longitud de tour media obtenida
%   
%   INPUTS:
%
%       numTests : número de ejecuciones del algoritmo
%       ... : veáse parámetros de la función ACS
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
        [bestLength, ~] = ACS(distMatrix, beta, rho, xi, q0, numAnts, maxIters, maxTime);
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
            opacity = (pheromones(i,j) - minP)/max_minP;
%             opacity = min(1, opacity*2);
            plot(coords([i, j], 1), coords([i, j], 2), 'b-', 'LineWidth', 3,'Color',[1,0,0,opacity])
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
