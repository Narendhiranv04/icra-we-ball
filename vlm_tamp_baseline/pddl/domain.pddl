(define (domain vlm-tamp-kitchen)
  (:requirements :strips :negative-preconditions :action-costs)
  (:predicates
    (Movable ?o) (Receptacle ?o) (Region ?r) (Workspace ?w)
    (AtRobot ?w) (At ?o ?l)
    (RequiresWorkspace ?l ?w)
    (Accessible ?r) (Open ?r) (Closed ?r) (Inspected ?r)
    (HandEmpty) (Holding ?o)
    (Poured ?source ?target) (Stirred ?tool ?target)
    (CanMove ?from ?to ?control)
    (CanInspect ?region ?workspace ?control)
    (CanPick ?object ?location ?workspace ?grasp ?control)
    (CanPickObject ?object ?target ?location ?workspace ?grasp ?control)
    (CanPlace ?object ?region ?workspace ?pose ?control)
    (CanPlaceObject ?object ?target ?location ?workspace ?pose ?control)
    (CanPour ?source ?target ?location ?workspace ?control)
    (CanStir ?tool ?target ?location ?workspace ?control))

  (:functions (total-cost))

  (:action move
    :parameters (?from ?to ?control)
    :precondition (and (AtRobot ?from) (CanMove ?from ?to ?control))
    :effect (and (AtRobot ?to) (not (AtRobot ?from))
                 (increase (total-cost) 1)))

  (:action inspect-closed
    :parameters (?region ?workspace ?control)
    :precondition (and (Region ?region) (AtRobot ?workspace) (HandEmpty)
                       (Closed ?region)
                       (RequiresWorkspace ?region ?workspace)
                       (CanInspect ?region ?workspace ?control))
    :effect (and (Open ?region) (Accessible ?region) (Inspected ?region)
                 (not (Closed ?region)) (increase (total-cost) 1)))

  (:action inspect-open
    :parameters (?region ?workspace ?control)
    :precondition (and (Region ?region) (AtRobot ?workspace) (HandEmpty)
                       (Open ?region)
                       (RequiresWorkspace ?region ?workspace)
                       (CanInspect ?region ?workspace ?control))
    :effect (and (Accessible ?region) (Inspected ?region)
                 (increase (total-cost) 1)))

  (:action pick
    :parameters (?object ?location ?workspace ?grasp ?control)
    :precondition (and (Movable ?object) (At ?object ?location)
                       (Accessible ?location) (AtRobot ?workspace) (HandEmpty)
                       (RequiresWorkspace ?location ?workspace)
                       (CanPick ?object ?location ?workspace ?grasp ?control))
    :effect (and (Holding ?object) (not (HandEmpty))
                 (not (At ?object ?location)) (increase (total-cost) 1)))

  (:action pick-object
    :parameters (?object ?target ?location ?workspace ?grasp ?control)
    :precondition (and (Movable ?object) (Receptacle ?target)
                       (At ?object ?target) (At ?target ?location)
                       (Accessible ?location) (AtRobot ?workspace) (HandEmpty)
                       (RequiresWorkspace ?location ?workspace)
                       (CanPickObject ?object ?target ?location ?workspace
                                      ?grasp ?control))
    :effect (and (Holding ?object) (not (HandEmpty))
                 (not (At ?object ?target)) (increase (total-cost) 1)))

  (:action place
    :parameters (?object ?region ?workspace ?pose ?control)
    :precondition (and (Movable ?object) (Region ?region) (Holding ?object)
                       (Accessible ?region) (AtRobot ?workspace)
                       (RequiresWorkspace ?region ?workspace)
                       (CanPlace ?object ?region ?workspace ?pose ?control))
    :effect (and (At ?object ?region) (HandEmpty) (not (Holding ?object))
                 (increase (total-cost) 1)))

  (:action place-object
    :parameters (?object ?target ?location ?workspace ?pose ?control)
    :precondition (and (Movable ?object) (Receptacle ?target)
                       (At ?target ?location) (Holding ?object)
                       (Accessible ?location) (AtRobot ?workspace)
                       (RequiresWorkspace ?location ?workspace)
                       (CanPlaceObject ?object ?target ?location ?workspace
                                       ?pose ?control))
    :effect (and (At ?object ?target) (HandEmpty) (not (Holding ?object))
                 (increase (total-cost) 1)))

  (:action pour
    :parameters (?source ?target ?location ?workspace ?control)
    :precondition (and (Holding ?source) (Movable ?target) (Receptacle ?target)
                       (At ?target ?location) (Accessible ?location)
                       (AtRobot ?workspace)
                       (RequiresWorkspace ?location ?workspace)
                       (CanPour ?source ?target ?location ?workspace ?control))
    :effect (and (Poured ?source ?target) (increase (total-cost) 1)))

  (:action stir
    :parameters (?tool ?target ?location ?workspace ?control)
    :precondition (and (Holding ?tool) (Movable ?target) (Receptacle ?target)
                       (At ?target ?location) (Accessible ?location)
                       (AtRobot ?workspace)
                       (RequiresWorkspace ?location ?workspace)
                       (CanStir ?tool ?target ?location ?workspace ?control))
    :effect (and (Stirred ?tool ?target) (increase (total-cost) 1))))
