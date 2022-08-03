#ifndef TARGETHIT_HH
#define TARGETHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"

class G4AttDef;
class G4AttValue;

class TargetHit : public G4VHit
{
public:
    public:
    TargetHit();
    // TargetHit(G4int cellID);
    TargetHit(const TargetHit &right) = default;
    ~TargetHit() override;

    // TargetHit& operator=(const TargetHit &right) = default;
    // G4bool operator==(const TargetHit &right) const;

    inline void *operator new(size_t);
    inline void operator delete(void *aHit);

    // void Draw() override;
    // const std::map<G4String,G4AttDef>* GetAttDefs() const override;
    // std::vector<G4AttValue>* CreateAttValues() const override;
    // void Print() override;

    // void SetCellID(G4int z) { fCellID = z; }
    // G4int GetCellID() const { return fCellID; }

    void SetEdep(G4double de) { fEdep = de; }
    void AddEdep(G4double de) { fEdep += de; }
    G4double GetEdep() const { return fEdep; }

private:
    // G4int fCellID = -1;
    G4double fEdep = 0.;
};

using TargetHitsCollection = G4THitsCollection<TargetHit>;

extern G4ThreadLocal G4Allocator<TargetHit>* TargetHitAllocator;

inline void* TargetHit::operator new(size_t)
{
    if (!TargetHitAllocator) 
    {
        TargetHitAllocator = new G4Allocator<TargetHit>;
    }
    return (void*)TargetHitAllocator->MallocSingle();
}

inline void TargetHit::operator delete(void* aHit)
{
    TargetHitAllocator -> FreeSingle((TargetHit*) aHit);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif