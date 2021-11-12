
import React from "react";
import robo from "../src/Image/robot2.gif";
import Common from "./Common";
import Services from "./Service";
import Process from "./Process";
import About from "./About";
import Contact from "./Contact";
import Navbar from "./Navbar";
import Particles from "./Particles"

const Home = () => {
  return (
   
    <div>

<Navbar />
    <Particles>
      <Common
        name="IEMAIWRITER"
        imgsrc={robo}
        visit="/register"
        btname="Get Started"
      />
      </Particles>
{/* about section  */}
   <About/>
    <Process/>
		<Services/>
    <Contact/>
	
	 </div>
	 

  );
};

export default Home;


