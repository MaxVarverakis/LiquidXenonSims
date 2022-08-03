#include "construction.hh"

MyDetectorConstruction::MyDetectorConstruction()
{
    target = new G4GenericMessenger(this, "/target/", "Target Construction");

    target -> DeclareProperty("radLengths", n, "Number of radiation lengths");
    target -> DeclareProperty("xenon", liquidXenon, "Use liquid Xenon target");
    target -> DeclareProperty("window", windows, "Use Beryllium windows around target");
    
    liquidXenon = true;
    windows = true;

    targetPos = G4ThreeVector(0., 0., 0.);
    n = .250;
}

MyDetectorConstruction::~MyDetectorConstruction()
{
    delete target;
}

G4VPhysicalVolume *MyDetectorConstruction::Construct()
{
    G4NistManager *nist = G4NistManager::Instance();

    air = nist -> FindOrBuildMaterial("G4_Galactic");

    solidWorld = new G4Box("solidWorld", 0.5 * m, 0.5 * m, 0.5 * m);
    logicWorld = new G4LogicalVolume(solidWorld, air, "logicWorld");
    physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);

    ConstructTarget();

    return physWorld;
}

void MyDetectorConstruction::ConstructWindows(G4double targetWidth)
{
    G4double windowWidth = 147.395 * micrometer;
    G4double offset = targetWidth + windowWidth / 2.;
    G4NistManager *nist = G4NistManager::Instance();
    windowMaterial = nist -> FindOrBuildMaterial("G4_Be");

    // calculate where to place windows on z-axis
    G4double zIn = targetPos.z() - offset;
    G4double zOut = targetPos.z() + offset;

    solidWindowIn = new G4Tubs("solidWindowIn", 0., 10. * mm, windowWidth / 2., 0., twopi); // r: 0 -> 1 m, z: 0 -> 147 µm, phi: 0 -> 2 pi
    logicWindowIn = new G4LogicalVolume(solidWindowIn, windowMaterial, "logicWindowIn");
    new G4PVPlacement(0, G4ThreeVector(0., 0., zIn), logicWindowIn, "physWindowIn", logicWorld, false, 0, true);
    
    solidWindowOut = new G4Tubs("solidWindowOut", 0., 10. * mm, windowWidth / 2., 0., twopi); // r: 0 -> 1 m, z: 0 -> 147 µm, phi: 0 -> 2 pi
    logicWindowOut = new G4LogicalVolume(solidWindowOut, windowMaterial, "logicWindowOut");
    new G4PVPlacement(0, G4ThreeVector(0., 0., zOut), logicWindowOut, "physWindowOut", logicWorld, false, 0, true);
}

void MyDetectorConstruction::ConstructTarget()
{
    G4NistManager *nist = G4NistManager::Instance();

    if (liquidXenon)
    {
        // L_RL = 2.872 * cm;

        // G4NistManager *nist = G4NistManager::Instance();
        // // liquid Xenon info: https://pdg.lbl.gov/2019/AtomicNuclearProperties/HTML/liquid_xenon_Xe.html
        // G4double lqdXe_density = 2.953 * g/cm3;
        // G4double A = 131.2930 * g/mole;  // atomic mass
        // G4double Z = 54.;  // atomic number

        // elXe = new G4Element("Xenon", "Xe", Z, A);
        // lqdXe = new G4Material("LiquidXe", lqdXe_density, 1);
        // lqdXe -> AddElement(elXe, 1.0);
        // targetMaterial = lqdXe;

        targetMaterial = nist -> FindOrBuildMaterial("G4_lXe");
    }
    else
    {
        // L_RL = 0.4094 * cm;

        // G4NistManager *nist = G4NistManager::Instance();
        targetMaterial = nist -> FindOrBuildMaterial("G4_Ta");
    }

    L_RL = targetMaterial -> GetRadlen();
    // G4cout << "L_RL: " << L_RL / cm << " cm" << G4endl;
    dLRL = n * L_RL;

    solidTarget = new G4Box("solidTarget", 0.25 * m, 0.25 * m, dLRL);
    logicTarget = new G4LogicalVolume(solidTarget, targetMaterial, "logicTarget");
    physTarget = new G4PVPlacement(0, targetPos, logicTarget, "physTarget", logicWorld, false, 0, true);

    // fScoringVolume = logicTarget;

    if (windows)
        ConstructWindows(dLRL);
}

void MyDetectorConstruction::ConstructSDandField()
{
    G4SDManager *SDman = G4SDManager::GetSDMpointer();

    SensitiveTarget *targ = new SensitiveTarget("/target");
    SensitiveWindow *bin = new SensitiveWindow("/BeIn");
    SensitiveWindow *bout = new SensitiveWindow("/BeOut");
    
    SDman -> AddNewDetector(targ);
    SDman -> AddNewDetector(bin);
    SDman -> AddNewDetector(bout);

    logicTarget -> SetSensitiveDetector(targ);
    logicWindowIn -> SetSensitiveDetector(bin);
    logicWindowOut -> SetSensitiveDetector(bout);
}