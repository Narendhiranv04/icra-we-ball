(define (stream vlm-tamp-kitchen)
  (:stream sample-motion
    :inputs (?from ?to)
    :domain (and (Workspace ?from) (Workspace ?to))
    :outputs (?control)
    :certified (CanMove ?from ?to ?control))

  (:stream sample-inspect
    :inputs (?region ?workspace)
    :domain (and (Region ?region) (Workspace ?workspace)
                 (RequiresWorkspace ?region ?workspace))
    :outputs (?control)
    :certified (CanInspect ?region ?workspace ?control))

  (:stream sample-pick
    :inputs (?object ?location ?workspace)
    :domain (and (Movable ?object) (Region ?location) (Workspace ?workspace)
                 (RequiresWorkspace ?location ?workspace))
    :outputs (?grasp ?control)
    :certified (CanPick ?object ?location ?workspace ?grasp ?control))

  (:stream sample-pick-object
    :inputs (?object ?target ?location ?workspace)
    :domain (and (Movable ?object) (Receptacle ?target)
                 (At ?target ?location) (Region ?location)
                 (Workspace ?workspace)
                 (RequiresWorkspace ?location ?workspace))
    :outputs (?grasp ?control)
    :certified (CanPickObject ?object ?target ?location ?workspace
                              ?grasp ?control))

  (:stream sample-place
    :inputs (?object ?region ?workspace)
    :domain (and (Movable ?object) (Region ?region) (Workspace ?workspace)
                 (RequiresWorkspace ?region ?workspace))
    :outputs (?pose ?control)
    :certified (CanPlace ?object ?region ?workspace ?pose ?control))

  (:stream sample-place-object
    :inputs (?object ?target ?location ?workspace)
    :domain (and (Movable ?object) (Receptacle ?target)
                 (At ?target ?location) (Region ?location)
                 (Workspace ?workspace)
                 (RequiresWorkspace ?location ?workspace))
    :outputs (?pose ?control)
    :certified (CanPlaceObject ?object ?target ?location ?workspace
                               ?pose ?control))

  (:stream sample-pour
    :inputs (?source ?target ?location ?workspace)
    :domain (and (Movable ?source) (Movable ?target) (Receptacle ?target)
                 (Region ?location) (Workspace ?workspace)
                 (RequiresWorkspace ?location ?workspace))
    :outputs (?control)
    :certified (CanPour ?source ?target ?location ?workspace ?control))

  (:stream sample-stir
    :inputs (?tool ?target ?location ?workspace)
    :domain (and (Movable ?tool) (Movable ?target) (Receptacle ?target)
                 (Region ?location) (Workspace ?workspace)
                 (RequiresWorkspace ?location ?workspace))
    :outputs (?control)
    :certified (CanStir ?tool ?target ?location ?workspace ?control)))
