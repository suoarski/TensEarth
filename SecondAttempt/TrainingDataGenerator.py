
import os
import h5py
import stripy
import meshplex
import numpy as np
import pyvista as pv
from gospl.model import Model as sim
from gospl._fortran import definegtin
from scipy.interpolate import griddata

class DataGenerator:
    
    #Initiate with various optional parameters that we can change
    def __init__(self,
                
                # Mesh and simulation parameters
                subdivisions = 6, 
                radius = 6378137,
                
                # Time parameters as required by YML file
                startTime = 0,
                endTime = 60000000,
                tOut = 1000000,
                deltaTime = 100000,
                tecTime = 1000000,
                
                # Noise parameters for initial topography and tectonic uplift
                topographyNoiseParameters = {
                    "octaves" : 8,
                    "octaveStepSize" : 1.4,
                    "amplitudeStepSize" : 2.0,
                    "initialFrequency" : 0.000001,
                    "noiseMinMax" : np.array([-4000, 1000])},
                upliftNoiseParameters = {
                    "octaves" : 6,
                    "octaveStepSize" : 1.4,
                    "amplitudeStepSize" : 2.4,
                    "initialFrequency" : 0.0000002,
                    "noiseMinMax" : np.array([0, 2000])},
                
                # Directory file name patterns
                dataDir = './Data/GosplTrials',
                npzDirFormat = '{}/NPZFiles',
                trialDirFormat = '{}/Trial{}',
                initElevNPZformat = '{}/initElevations.npz',
                upliftNPZformat = '{}/uplift.npz',
                tecVerticalNPZformat = '{}/tecvert.npz',
                h5FilesFormat = '{}/NoiseSphere/h5/',
                animationDir = 'ErosionAnimation.mp4',
                templateYMLfile = './Data/TemplateUpliftYML.yml',
                outputYMLfileName = 'GosplParameters.yml',
                
                amplificationFactor = 90,
                animationFramesPerIteration = 8,
                
                scalarsToInterpolate=['elev', 'erodep', 'flowAcc'],
                uvSphereResolution = [512, 258]
                ):
        
        # Mesh parameters
        self.subdivisions = subdivisions
        self.radius = radius
        
        # Time Parameters
        self.startTime = startTime
        self.endTime = endTime
        self.tOut = tOut
        self.deltaTime = deltaTime
        self.tecTime = tecTime
        
        # Noise parameters
        self.topographyNoiseParameters = topographyNoiseParameters
        self.upliftNoiseParameters = upliftNoiseParameters
        
        # Directories
        self.dataDir = dataDir
        self.npzDirFormat = npzDirFormat
        self.trialDirFormat = trialDirFormat
        self.initElevNPZformat = initElevNPZformat  
        self.upliftNPZformat = upliftNPZformat
        self.tecVerticalNPZformat = tecVerticalNPZformat
        self.templateYMLfile = templateYMLfile
        self.outputYMLfileName = outputYMLfileName
        self.h5FilesFormat = h5FilesFormat
        self.animationDir = animationDir
        
        # Initiate Icosphere
        icosphere = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=self.subdivisions, include_face_points=False)
        self.XYZ = icosphere._points * self.radius
        self.icoFaces = self.stripyCellsToPyvistaFaces(icosphere.simplices)
        self.icoMesh = pv.PolyData(self.XYZ, self.icoFaces)
        self.icoMesh['cells'] = icosphere.simplices
        
        # Generate initial random noise to represent initial elevations and tectonic uplifts
        self.elevations = self.sampleNoise(self.topographyNoiseParameters)
        self.uplift = self.sampleNoise(self.upliftNoiseParameters)
        self.icoMesh['elevations'] = self.elevations
        self.icoMesh['uplift'] = self.uplift
        
        # Properties for visualizations
        self.amplificationFactor = amplificationFactor # How much to 'exagerate' terrain by
        self.animationFramesPerIteration = animationFramesPerIteration
        
        self.scalarsToInterpolate = scalarsToInterpolate
        self.uvSphereResolution = uvSphereResolution
        self.uvSphere = pv.Sphere(theta_resolution=uvSphereResolution[0], phi_resolution=uvSphereResolution[1])
    
    
    def getInitialTopography(self):
        fileDir = self.initElevNPZformat.format(self.npzDir)
        topoData = np.load(fileDir)
        return topoData['z']
    
    def getUpliftMap(self):
        fileDir = self.tecVerticalNPZformat.format(self.npzDir)
        upliftData = np.load(fileDir)
        return upliftData['z']
    
    #Create lon/lat to interpolate onto
    def getUVLonLat(self):
        _, uvLon, uvLat = self.cartesianToPolarCoords(self.uvSphere.points)
        uvLon[uvLon<=0] += 360
        return np.array([uvLon, uvLat]).T
    
    def getIcoLonLat(self):
        _, icoLon, icoLat = self.cartesianToPolarCoords(self.XYZ)
        icoLon[icoLon<=0] += 360
        return np.array([icoLon, icoLat]).T
    
    def interpolateGosplDataToUVsphere(self, finalIteration=60):
        interpolatedScalars = []
        icoLonLat = self.getIcoLonLat()
        uvLonLat = self.getUVLonLat()
        gosplDict = self.readGosplFile(finalIteration)
        
        #Loop through all scalars to interpolate
        for scalar in self.scalarsToInterpolate:
            newScalar = griddata(icoLonLat, gosplDict[scalar], uvLonLat)
            whereNAN = np.argwhere(np.isnan(newScalar))
            newScalar[whereNAN] = griddata(icoLonLat, gosplDict[scalar], uvLonLat[whereNAN], method='nearest')
            interpolatedScalars.append(newScalar)
        return np.array(interpolatedScalars)[:, :, 0]
    
    def interpolateInputDataToUVsphere(self, finalIteration=60):
        interpolatedScalars = []
        icoLonLat = self.getIcoLonLat()
        uvLonLat = self.getUVLonLat()
        
        initTopo = self.getInitialTopography()
        uvTopo = griddata(icoLonLat, initTopo, uvLonLat)
        whereNAN = np.argwhere(np.isnan(uvTopo))
        uvTopo[whereNAN] = griddata(icoLonLat, initTopo, uvLonLat[whereNAN], method='nearest')
        
        initUplift = self.getUpliftMap()
        uvUplift = griddata(icoLonLat, initUplift, uvLonLat)
        whereNAN = np.argwhere(np.isnan(uvUplift))
        uvUplift[whereNAN] = griddata(icoLonLat, initUplift, uvLonLat[whereNAN], method='nearest')
        return np.stack((uvTopo, uvUplift))
        
        
        
    
    
    #=============================================== Running and Loading Gospl Trials ======================================================
    # This function will prepare and run an entire gospl simulation based on parameters set for this object instance
    def prepareAndRunGOSPLtrial(self):
        self.createTrialDirectory()
        self.createInitialTopographyFile()
        self.createUpliftNPZfile()
        self.createYMLfile()
        self.runGOSPL()
    
    # Load data from a gospl trial that we have already run
    def loadTrial(self, trialNumber):
        self.trialDir = self.trialDirFormat.format(self.dataDir, trialNumber)
        self.npzDir = self.npzDirFormat.format(self.trialDir)
        
    # Save data in appropriate file directory
    def createTrialDirectory(self):
        if not os.path.isdir(self.dataDir):
            os.mkdir(self.dataDir)
        
        #Create a new subdirectory for the newest trial
        trialNumber = 0
        self.trialDir = self.trialDirFormat.format(self.dataDir, trialNumber)
        while os.path.isdir(self.trialDir):
            trialNumber += 1
            self.trialDir = self.trialDirFormat.format(self.dataDir, trialNumber)
        self.npzDir = self.npzDirFormat.format(self.trialDir)
        initElevNPZ = self.initElevNPZformat.format(self.npzDir)
        os.mkdir(self.trialDir)
        os.mkdir(self.npzDir)
    
    # Creates the NPZ file of initial topography for GOSPL
    def createInitialTopographyFile(self):
        vertices = self.XYZ
        cells = self.icoMesh['cells']
        neighbs = self.getNeighbourIds(vertices, cells)
        elevations = self.elevations
        fileDir = self.initElevNPZformat.format(self.npzDir)
        np.savez_compressed(fileDir, v=vertices, c=cells, n=neighbs.astype(int), z=elevations)
    
    # Create NPZ file of tectonic uplift
    def createUpliftNPZfile(self):
        uplift = self.uplift
        upliftFileDir = self.upliftNPZformat.format(self.npzDir)
        tecVertFileDir = self.tecVerticalNPZformat.format(self.npzDir)
        r, lon, lat = self.cartesianToPolarCoords(self.XYZ)
        movedXYZ = self.polarToCartesian(uplift/10000000, lon, lat)
        np.savez_compressed(upliftFileDir, xyz=movedXYZ)
        np.savez_compressed(tecVertFileDir, z=uplift/10000000)
    
    # Create and format YML file for this GOSPL trial
    def createYMLfile(self):
        with open(self.templateYMLfile, 'r') as ymlFile:
            ymlContent = ymlFile.read() 
        ymlContent = ymlContent.format(
                self.trialDir, 
                self.startTime,
                self.endTime,
                self.tOut,
                self.deltaTime,
                self.tecTime,
                self.startTime,
                self.endTime,
                self.trialDir, 
                self.trialDir, 
                self.trialDir)
        self.newYMLfileDir = self.trialDir + '/' + self.outputYMLfileName
        with open(self.newYMLfileDir, 'w') as newYMLfile:
            newYMLfile.write(ymlContent)
    
    def runGOSPL(self):
        mod = sim(self.newYMLfileDir, False, False)
        mod.runProcesses()
        mod.destroy()
    
    
    
    
    
    # ========================================================= Reading and Visualizing Data ========================================
    #The Gospl file contains simulation output data at particular iterations during the simulation
    def readGosplFile(self, iteration):
        gosplDict = {}
        fileDir = self.trialDir + '/NoiseSphere/h5/gospl.{}.p0.h5'.format(iteration)
        with h5py.File(fileDir, "r") as f:
            for key in f.keys():
                gosplDict[key] = np.array(f[key])
        return gosplDict
    
    #Exagerated mesh for plotting
    def createExageratedMesh(self, dataDict):
        r, lon, lat = self.cartesianToPolarCoords(self.XYZ)
        exegeratedRadius = self.radius + self.amplificationFactor * dataDict['elev']
        exageratedXYZ = self.polarToCartesian(exegeratedRadius[:, 0], lon, lat)
        exegeratedMesh = pv.PolyData(exageratedXYZ, self.icoFaces)
        for key in dataDict.keys():
            exegeratedMesh[key] = dataDict[key]
        return exegeratedMesh
    
    # Animate results from gospl simulation
    def animateGosplResults(self, iterations=10, scalarToPlot='elev'):
        
        #Create initial planet mesh
        dataDict = self.readGosplFile(0)
        planetMesh = self.createExageratedMesh(dataDict)
        
        #Initiate the plotter for animations
        plotter = pv.Plotter()
        plannetActor = plotter.add_mesh(planetMesh, scalars=dataDict[scalarToPlot])
        plotter.camera.zoom(1.4)
        plotter.open_movie(self.animationDir)
        for i in range(self.animationFramesPerIteration):
            plotter.write_frame()
        
        #Loop through all iteration data files and draw animation frames
        for i in range(iterations):
            dataDict = self.readGosplFile(i+1)
            planetMesh = self.createExageratedMesh(dataDict)
            newScalars = dataDict[scalarToPlot]

            #Draw animation frames
            plotter.remove_actor(plannetActor)
            plannetActor = plotter.add_mesh(planetMesh, scalars=dataDict[scalarToPlot])
            for i in range(self.animationFramesPerIteration):
                plotter.write_frame()
        plotter.close()
    
    
    
    
    
    # ========================================= Data Processing Functions ===========================================================
    # Given a list of XYZ coordinates, we sample a noise value at each point
    @staticmethod
    def samplePerlinNoise(XYZ, amplitude=1, frequency=4, offset=None):
        if offset == None: #Random offset if none other is specified
            offset = np.random.rand(3) * 100000
        freq = (frequency, frequency, frequency)
        noise = pv.perlin_noise(amplitude, freq, offset)
        return np.array([noise.EvaluateFunction(xyz) for xyz in XYZ])
    
    #Sample noise with multiple noise frequencies
    def sampleNoise(self, noiseParameters):
        noiseSum = np.zeros(self.XYZ.shape[0])
        for i in range(noiseParameters["octaves"]):
            frequency =  noiseParameters["initialFrequency"] * noiseParameters["octaveStepSize"] ** i
            noiseSum += self.samplePerlinNoise(self.XYZ, frequency=frequency) / (noiseParameters["amplitudeStepSize"]*(i+1))
        
        # Normalise noise and bring to desired noise range
        minMax = noiseParameters["noiseMinMax"]
        noiseSum -= np.min(noiseSum)
        noiseSum = noiseSum * (minMax[1] - minMax[0]) / np.max(noiseSum) + minMax[0]
        return noiseSum
    
    #Create list of neighbour ids, based on bfModel notebook tutorial
    @staticmethod
    def getNeighbourIds(icoXYZ, icoCells):
        Gmesh = meshplex.MeshTri(icoXYZ, icoCells)
        s = Gmesh.idx_hierarchy.shape
        a = np.sort(Gmesh.idx_hierarchy.reshape(s[0], -1).T)
        Gmesh.edges = {'points': np.unique(a, axis=0)}
        ngbNbs, ngbID = definegtin(len(icoXYZ), Gmesh.cells('points'), Gmesh.edges['points'])
        ngbIDs = ngbID[:,:8].astype(int)
        return ngbIDs
    
    #Create an array for the pyvista faces based on stripy cells
    @staticmethod
    def stripyCellsToPyvistaFaces(cells):
        faces = []
        for cell in cells:
            faces.append(3)
            faces.append(cell[0])
            faces.append(cell[1])
            faces.append(cell[2])
        return np.array(faces)
    
    #Coordinate transformation from spherical polar to cartesian
    @staticmethod
    def polarToCartesian(radius, theta, phi, useLonLat=True):
        if useLonLat == True:
            theta, phi = np.radians(theta+180.), np.radians(90. - phi)
        X = radius * np.cos(theta) * np.sin(phi)
        Y = radius * np.sin(theta) * np.sin(phi)
        Z = radius * np.cos(phi)
        
        #Return data either as a list of XYZ coordinates or as a single XYZ coordinate
        if (type(X) == np.ndarray):
            return np.stack((X, Y, Z), axis=1)
        else:
            return np.array([X, Y, Z])

    #Coordinate transformation from cartesian to polar
    @staticmethod
    def cartesianToPolarCoords(XYZ, useLonLat=True):
        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
        R = (X**2 + Y**2 + Z**2)**0.5
        theta = np.arctan2(Y, X)
        phi = np.arccos(Z / R)

        #Return results either in spherical polar or leave it in radians
        if useLonLat == True:
            theta, phi = np.degrees(theta), np.degrees(phi)
            lon, lat = theta - 180, 90 - phi
            lon[lon < -180] = lon[lon < -180] + 360
            return R, lon, lat
        else:
            return R, theta, phi
    