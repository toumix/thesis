class Object:
    ...

class Arrow:
    dom : Object
    cod : Object

    @staticmethod
    def id(x : Object) -> Arrow[x, x]:
        ...

    def then(self, other : Arrow[self.cod, y])
            -> Arrow[self.dom, y]:
        ...

class Functor:
    @overload
    def __call__(self, x : Object) -> Object:
        ...

    @overload
    def __call__(self, f : Arrow[x, y])
            -> Arrow[self(x), self(y)]:
        ...

class Transformation:
    dom : Functor
    cod : Functor

    def __call__(self, x : Object)
            -> Arrow[self.dom(x), self.cod(x)]:
        ...
