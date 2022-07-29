#include "detector.hh"

MySensitiveDetector::MySensitiveDetector(G4String name) : G4VSensitiveDetector(name)
{
    // count = 0;
    // Etot = 0.;
}

MySensitiveDetector::~MySensitiveDetector()
{}

G4bool MySensitiveDetector::ProcessHits(G4Step *step, G4TouchableHistory *R0history)
{
    G4Track *track = step -> GetTrack();
    // track -> SetTrackStatus(fStopAndKill);

    G4String particleName = track -> GetDefinition() -> GetParticleName();

    G4StepPoint *preStepPoint = step -> GetPreStepPoint();
    G4StepPoint *postStepPoint = step -> GetPostStepPoint();
    
    G4AnalysisManager *man = G4AnalysisManager::Instance();

    // nonzero energy particles only
    // if(energy)
    //     G4cout << particleName << " energy: " << energy << G4endl;
    
    // G4cout << particleName << " energy: " << energy << G4endl;
    

    // if (preStepPoint -> GetStepStatus() == fGeomBoundary)
    // {
    //     G4double energy = preStepPoint -> GetKineticEnergy();
        // G4cout << "Incident Energy: " << energy << G4endl;
        // G4double mass = preStepPoint -> GetMass();
        // G4cout << "Incident Mass: " << mass << G4endl;
    //     G4ThreeVector pos = preStepPoint -> GetPosition();
    //     G4double traverseWidth = pos.getRho();
    //     G4double angle = pos.getTheta();

    //     man -> FillNtupleDColumn(0, energy);
    //     man -> FillNtupleDColumn(1, traverseWidth);
    //     man -> FillNtupleDColumn(2, angle);
    //     man -> AddNtupleRow(0);
    // }
    // if (particleName == "e-" || particleName == "e+")
    if (postStepPoint -> GetStepStatus() == fGeomBoundary)
    {
        G4double energy = postStepPoint -> GetKineticEnergy();
        // Etot += energy;
        // G4cout << "Total Energy: " << Etot << G4endl;
        // G4cout << "Total Energy: " << G4BestUnit(Etot, "Energy") << G4endl;
        if (particleName == "e+")
        {
            // man -> FillNtupleDColumn(0, 0, energy);
            G4ThreeVector pos = postStepPoint -> GetPosition();
            G4double traverseWidth = pos.getRho();
            G4double azimuthal = pos.getTheta();
            G4double angle = pos.getPhi();

            G4LorentzVector p4 = track -> GetDynamicParticle() -> Get4Momentum();
            G4double invariant = p4.m();

            // G4cout << "Invariant: " << G4BestUnit(invariant, "Energy") << G4endl;
            // G4cout << p4.m() << G4endl;
            // G4cout << p4.px() << ' ' << p4[0] << ' ' << p4.e() << ' ' << p4[3] << G4endl;
            // G4ThreeVector momentumDir = postStepPoint -> GetMomentumDirection();
            // G4double mass = postStepPoint -> GetMass();
            // G4cout << mass << G4endl;
            // G4cout << energy << p4[3] << G4endl;
            // G4double momentum = std::sqrt(energy * energy + 2.0 * mass * energy);
            // G4LorentzVector p4 = G4LorentzVector(momentumDir.x() * momentum,
            //                                         momentumDir.y() * momentum, 
            //                                         momentumDir.z() * momentum, 
            //                                         energy + mass);
            // G4cout << p4.m() << G4endl;
            // G4cout << "Momentum Four Vector: " << p4 << ", Mass: " << mass 
            // << ", Energy: " << energy << G4endl;
            // G4LorentzVector fourVec = postStepPoint -> Get4Momentum();
            
            // G4cout << p4[0] << G4endl;
            // G4cout << G4BestUnit(p4[0], "Momentum") << G4endl;
            // G4cout << G4BestUnit(p4[1], "Momentum") << G4endl;
            // G4cout << G4BestUnit(p4[2], "Momentum") << G4endl;
            // G4cout << G4BestUnit(p4[3], "Energy") << G4endl;
            // G4cout << p4[3] << G4endl;
            
            // G4cout << p4 << G4endl;
            // G4cout << P << G4endl;

            man -> FillNtupleDColumn(0, 0, energy);
            man -> FillNtupleDColumn(0, 1, traverseWidth);
            man -> FillNtupleDColumn(0, 2, azimuthal);
            man -> FillNtupleDColumn(0, 3, angle);
            man -> FillNtupleDColumn(0, 4, p4.px());
            man -> FillNtupleDColumn(0, 5, p4.py());
            man -> FillNtupleDColumn(0, 6, p4.pz());
            man -> FillNtupleDColumn(0, 7, p4.e());
            man -> FillNtupleDColumn(0, 8, invariant);
            man -> AddNtupleRow(0);

            // man -> FillNtupleDColumn(2, 0, p4.px());
            // man -> FillNtupleDColumn(2, 1, p4.py());
            // man -> FillNtupleDColumn(2, 2, p4.pz());
            // man -> FillNtupleDColumn(2, 3, p4.e());
            // man -> FillNtupleDColumn(2, 4, invariant);
            // man -> AddNtupleRow(2);

            // std::ofstream outfile;
            // outfile.open("output.txt", std::ios::app);
            // outfile << radDistance << G4endl;
            // outfile.close();

            // const G4VTouchable *touchable = postStepPoint -> GetTouchable();
            // G4int copyNo = touchable -> GetCopyNumber();

            // G4cout << particleName << " position: " << pos << " cm, radial distance: " << radDistance << " cm, energy: " << energy << " GeV" << G4endl;
            // G4cout << radDistance << G4endl;

            // counter++;

            // G4int evt = G4RunManager::GetRunManager() -> GetCurrentEvent() -> GetEventID();

            // man -> FillNtupleIColumn(0, evt);
            
            // man -> FillNtupleDColumn(3, energy);
            // man -> FillNtupleDColumn(4, radDistance);
            // man -> FillNtupleDColumn(5, pos[0]);
            // man -> FillNtupleDColumn(6, pos[1]);
            // man -> FillNtupleDColumn(7, pos[2]);

            // if (particleName == "e-")
            // {
            //     man -> FillNtupleDColumn(0, 0, energy);
            //     man -> FillNtupleDColumn(0, 1, traverseWidth);
            //     man -> FillNtupleDColumn(0, 2, angle);
            //     man -> AddNtupleRow(0);
            // }
            // else if (particleName == "e+")
            // {
            //     man -> FillNtupleDColumn(1, 0, energy);
            //     man -> FillNtupleDColumn(1, 1, traverseWidth);
            //     man -> FillNtupleDColumn(1, 2, angle);
            //     man -> AddNtupleRow(1);
            // }
        }
        // G4cout << counter << G4endl;
    }
}
