#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4GeneralParticleSource.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4GenericMessenger.hh"
#include "G4UImanager.hh"

class MyPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
    MyPrimaryGenerator();
    ~MyPrimaryGenerator();
    
    virtual void GeneratePrimaries(G4Event*);

private:
    // Messenger doesn't really do anything, need to restart sim to change beam distribution (change from here)
    G4GenericMessenger *fMessenger;
    G4bool isBeamDist = false;

    G4GeneralParticleSource *fBeamGun;
    G4ParticleGun *fParticleGun;
};

#endif